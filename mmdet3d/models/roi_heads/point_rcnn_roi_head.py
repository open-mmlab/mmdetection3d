import torch
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class PointRCNNROIHead(Base3DRoIHead):
    """PointRCNN roi head for PointRCNN.

    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 bbox_head,
                 num_classes=3,
                 depth_normalizer=70,
                 point_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None):
        super(PointRCNNROIHead, self).__init__(
            bbox_head=bbox_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.num_classes = num_classes
        self.depth_normalizer = depth_normalizer

        if point_roi_extractor is not None:
            self.point_roi_extractor = build_roi_extractor(point_roi_extractor)

        self.init_assigner_sampler()

    def init_weights(self, pretrained):
        pass

    def init_mask_head(self):
        """Initialize mask head, skip since ``PointRCNNROIHead`` does not have
        one."""
        pass

    def init_bbox_head(self, bbox_head):
        """Initialize box head."""
        self.bbox_head = build_head(bbox_head)

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

    def forward_train(self, feats_dict, point_scores, img_metas, proposal_list,
                      gt_bboxes_3d, gt_labels_3d):
        """Training forward function of PartAggregationROIHead.

        Args:
            feats_dict (dict): Contains features from the first stage.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.
                The dictionary should contain the following keys:

                - boxes_3d (:obj:`BaseInstance3DBoxes`): Proposal bboxes
                - labels_3d (torch.Tensor): Labels of proposals
                - cls_preds (torch.Tensor): Original scores of proposals
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels_3d (list[LongTensor]): GT labels of each sample.

        Returns:
            dict: losses from each head.

                - loss_semantic (torch.Tensor): loss of semantic head
                - loss_bbox (torch.Tensor): loss of bboxes
        """
        losses = dict()
        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)

        features = feats_dict['fp_features'][-1]
        points = feats_dict['fp_xyz'][-1]
        features = features.transpose(1, 2).contiguous()
        point_depths = points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]
        features = torch.cat(features_list, dim=2)

        bbox_results = self._bbox_forward_train(features, points,
                                                sample_results)
        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(self, feats_dict, point_scores, img_metas, proposal_list,
                    **kwargs):
        """Simple testing forward function of PartAggregationROIHead.

        Note:
            This function assumes that the batch size is 1

        Args:
            feats_dict (dict): Contains features from the first stage.
            obj_scores (dict): Contains object scores from the first stage.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.

        Returns:
            dict: Bbox results of one frame.
        """
        rois = bbox3d2roi([res['boxes_3d'].tensor for res in proposal_list])
        labels_3d = [res['labels_3d'] for res in proposal_list]
        cls_preds = [res['scores_3d'] for res in proposal_list]

        features = feats_dict['fp_features'][-1]
        points = feats_dict['fp_xyz'][-1]
        features = features.transpose(1, 2).contiguous()
        point_depths = points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]

        features = torch.cat(features_list, dim=2)
        batch_size = features.shape[0]
        bbox_results = self._bbox_forward(features, points, batch_size, rois)

        bbox_list = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            labels_3d,
            cls_preds,
            img_metas,
            cfg=self.test_cfg)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _bbox_forward_train(self, global_feats, local_feats, sampling_results):
        """Forward training function of roi_extractor and bbox_head.

        Args:
            global_feats (torch.Tensor): global Point-wise semantic features.
            local_feats (torch.Tensor): local Point-wise semantic features.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        batch_size = global_feats.shape[0]
        bbox_results = self._bbox_forward(global_feats, local_feats,
                                          batch_size, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, global_feats, points, batch_size, rois):
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            global_feats (torch.Tensor): Point-wise semantic features.
            local_feats (torch.Tensor): Point-wise part prediction features.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_point_feats = self.point_roi_extractor(global_feats, points,
                                                      batch_size, rois)
        cls_score, bbox_pred = self.bbox_head(pooled_point_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

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
