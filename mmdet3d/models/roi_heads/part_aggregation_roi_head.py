# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class PartAggregationROIHead(Base3DRoIHead):
    """Part aggregation roi head for PartA2.

    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        part_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 semantic_head,
                 num_classes=3,
                 seg_roi_extractor=None,
                 part_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PartAggregationROIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        assert semantic_head is not None
        self.semantic_head = build_head(semantic_head)

        if seg_roi_extractor is not None:
            self.seg_roi_extractor = build_roi_extractor(seg_roi_extractor)
        if part_roi_extractor is not None:
            self.part_roi_extractor = build_roi_extractor(part_roi_extractor)

        self.init_assigner_sampler()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def init_mask_head(self):
        """Initialize mask head, skip since ``PartAggregationROIHead`` does not
        have one."""
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

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def forward_train(self, feats_dict, voxels_dict, img_metas, proposal_list,
                      gt_bboxes_3d, gt_labels_3d):
        """Training forward function of PartAggregationROIHead.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxels_dict (dict): Contains information of voxels.
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
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict['seg_features'], voxels_dict, gt_bboxes_3d,
                gt_labels_3d)
            losses.update(semantic_results['loss_semantic'])

        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                feats_dict['seg_features'], semantic_results['part_feats'],
                voxels_dict, sample_results)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(self, feats_dict, voxels_dict, img_metas, proposal_list,
                    **kwargs):
        """Simple testing forward function of PartAggregationROIHead.

        Note:
            This function assumes that the batch size is 1

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxels_dict (dict): Contains information of voxels.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.

        Returns:
            dict: Bbox results of one frame.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert self.with_semantic

        semantic_results = self.semantic_head(feats_dict['seg_features'])

        rois = bbox3d2roi([res['boxes_3d'].tensor for res in proposal_list])
        labels_3d = [res['labels_3d'] for res in proposal_list]
        cls_preds = [res['cls_preds'] for res in proposal_list]
        bbox_results = self._bbox_forward(feats_dict['seg_features'],
                                          semantic_results['part_feats'],
                                          voxels_dict, rois)

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

    def _bbox_forward_train(self, seg_feats, part_feats, voxels_dict,
                            sampling_results):
        """Forward training function of roi_extractor and bbox_head.

        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.
            voxels_dict (dict): Contains information of voxels.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(seg_feats, part_feats, voxels_dict,
                                          rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, seg_feats, part_feats, voxels_dict, rois):
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.
            voxels_dict (dict): Contains information of voxels.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_seg_feats = self.seg_roi_extractor(seg_feats,
                                                  voxels_dict['voxel_centers'],
                                                  voxels_dict['coors'][..., 0],
                                                  rois)
        pooled_part_feats = self.part_roi_extractor(
            part_feats, voxels_dict['voxel_centers'],
            voxels_dict['coors'][..., 0], rois)
        cls_score, bbox_pred = self.bbox_head(pooled_seg_feats,
                                              pooled_part_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            pooled_seg_feats=pooled_seg_feats,
            pooled_part_feats=pooled_part_feats)
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

    def _semantic_forward_train(self, x, voxels_dict, gt_bboxes_3d,
                                gt_labels_3d):
        """Train semantic head.

        Args:
            x (torch.Tensor): Point-wise semantic features for segmentation
            voxels_dict (dict): Contains information of voxels.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels

        Returns:
            dict: Segmentation results including losses
        """
        semantic_results = self.semantic_head(x)
        semantic_targets = self.semantic_head.get_targets(
            voxels_dict, gt_bboxes_3d, gt_labels_3d)
        loss_semantic = self.semantic_head.loss(semantic_results,
                                                semantic_targets)
        semantic_results.update(loss_semantic=loss_semantic)
        return semantic_results
