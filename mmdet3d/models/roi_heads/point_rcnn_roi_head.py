# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class PointRCNNRoIHead(Base3DRoIHead):
    """RoI head for PointRCNN.

    Args:
        bbox_head (dict): Config of bbox_head.
        point_roi_extractor (dict): Config of RoI extractor.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        depth_normalizer (float, optional): Normalize depth feature.
            Defaults to 70.0.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 bbox_head,
                 point_roi_extractor,
                 train_cfg,
                 test_cfg,
                 depth_normalizer=70.0,
                 pretrained=None,
                 init_cfg=None):
        super(PointRCNNRoIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.depth_normalizer = depth_normalizer

        if point_roi_extractor is not None:
            self.point_roi_extractor = build_roi_extractor(point_roi_extractor)

        self.init_assigner_sampler()

    def init_bbox_head(self, bbox_head):
        """Initialize box head.

        Args:
            bbox_head (dict): Config dict of RoI Head.
        """
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self):
        """Initialize maek head."""
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    build_assigner(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)

    def forward_train(self, feats_dict, input_metas, proposal_list,
                      gt_bboxes_3d, gt_labels_3d):
        """Training forward function of PointRCNNRoIHead.

        Args:
            feats_dict (dict): Contains features from the first stage.
            imput_metas (list[dict]): Meta info of each input.
            proposal_list (list[dict]): Proposal information from rpn.
                The dictionary should contain the following keys:

                - boxes_3d (:obj:`BaseInstance3DBoxes`): Proposal bboxes
                - labels_3d (torch.Tensor): Labels of proposals
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels_3d (list[LongTensor]): GT labels of each sample.

        Returns:
            dict: Losses from RoI RCNN head.
                - loss_bbox (torch.Tensor): Loss of bboxes
        """
        features = feats_dict['features']
        points = feats_dict['points']
        point_cls_preds = feats_dict['points_cls_preds']
        sem_scores = point_cls_preds.sigmoid()
        point_scores = sem_scores.max(-1)[0]
        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)

        # concat the depth, semantic features and backbone features
        features = features.transpose(1, 2).contiguous()
        point_depths = points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]
        features = torch.cat(features_list, dim=2)

        bbox_results = self._bbox_forward_train(features, points,
                                                sample_results)
        losses = dict()
        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(self, feats_dict, img_metas, proposal_list, **kwargs):
        """Simple testing forward function of PointRCNNRoIHead.

        Note:
            This function assumes that the batch size is 1

        Args:
            feats_dict (dict): Contains features from the first stage.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.

        Returns:
            dict: Bbox results of one frame.
        """
        rois = bbox3d2roi([res['boxes_3d'].tensor for res in proposal_list])
        labels_3d = [res['labels_3d'] for res in proposal_list]

        features = feats_dict['features']
        points = feats_dict['points']
        point_cls_preds = feats_dict['points_cls_preds']
        sem_scores = point_cls_preds.sigmoid()
        point_scores = sem_scores.max(-1)[0]

        features = features.transpose(1, 2).contiguous()
        point_depths = points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]

        features = torch.cat(features_list, dim=2)
        batch_size = features.shape[0]
        bbox_results = self._bbox_forward(features, points, batch_size, rois)
        object_score = bbox_results['cls_score'].sigmoid()
        bbox_list = self.bbox_head.get_bboxes(
            rois,
            object_score,
            bbox_results['bbox_pred'],
            labels_3d,
            img_metas,
            cfg=self.test_cfg)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _bbox_forward_train(self, features, points, sampling_results):
        """Forward training function of roi_extractor and bbox_head.

        Args:
            features (torch.Tensor): Backbone features with depth and \
                semantic features.
            points (torch.Tensor): Pointcloud.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        batch_size = features.shape[0]
        bbox_results = self._bbox_forward(features, points, batch_size, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, features, points, batch_size, rois):
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            features (torch.Tensor): Backbone features with depth and
                semantic features.
            points (torch.Tensor): Pointcloud.
            batch_size (int): Batch size.
            rois (torch.Tensor): RoI boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_point_feats = self.point_roi_extractor(features, points,
                                                      batch_size, rois)

        cls_score, bbox_pred = self.bbox_head(pooled_point_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    @torch.no_grad()
    def _assign_and_sample(self, proposal_list, gt_bboxes_3d, gt_labels_3d):
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[dict]): Proposals produced by RPN.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = proposal_list[batch_idx]
            cur_boxes = cur_proposal_list['boxes_3d']
            cur_labels_3d = cur_proposal_list['labels_3d']
            cur_gt_bboxes = gt_bboxes_3d[batch_idx].to(cur_boxes.device)
            cur_gt_labels = gt_labels_3d[batch_idx]
            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_boxes.tensor[pred_per_cls],
                        cur_gt_bboxes.tensor[gt_per_cls],
                        gt_labels=cur_gt_labels[gt_per_cls])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_boxes.tensor,
                    cur_gt_bboxes.tensor,
                    gt_labels=cur_gt_labels)

            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results
