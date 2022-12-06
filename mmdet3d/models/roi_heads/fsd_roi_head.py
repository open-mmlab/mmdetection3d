# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple

import torch
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from torch import Tensor
from torch.nn import functional as F

from mmdet.models.task_modules import AssignResult
from mmdet3d.structures.ops import bbox3d2result, bbox3d2roi
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.models.task_modules.builder import build_assigner, build_sampler
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead
from mmdet3d.registry import MODELS


@MODELS.register_module()
class GroupCorrectionHead(Base3DRoIHead):
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
                 num_classes=3,
                 roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes

        self.roi_extractor = build_roi_extractor(roi_extractor)

        self.init_assigner_sampler()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def init_mask_head(self):
        pass

    def init_bbox_head(self, bbox_roi_extractor: dict = None,
                       bbox_head: dict = None) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        # self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)
        self.bbox_head.train_cfg = self.train_cfg
        self.bbox_head.test_cfg = self.test_cfg

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

    def forward_train(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        img_metas,
        proposal_list,
        gt_bboxes_3d,
        gt_labels_3d
        ):

        losses = dict()

        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)

        bbox_results = self._bbox_forward_train(
            pts_xyz,
            pts_feats,
            pts_batch_idx,
            sample_results
        )

        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_inds,
        img_metas,
        proposal_list,
        gt_bboxes_3d,
        gt_labels_3d,
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


        assert len(proposal_list) == 1, 'only support bsz==1 to make cls_preds and labels_3d consistent with bbox_results'
        rois = bbox3d2roi([res[0].tensor for res in proposal_list])
        cls_preds = [res[1] for res in proposal_list]
        labels_3d = [res[2] for res in proposal_list]

        if len(rois) == 0:
            # fake prediction without velocity dims
            rois = torch.tensor([[0,0,0,5,1,1,1,0]], dtype=rois.dtype, device=rois.device)
            cls_preds = [torch.tensor([0.0], dtype=torch.float32, device=rois.device)]
            labels_3d = [torch.tensor([0], dtype=torch.int64, device=rois.device)]
           

        # cls_preds = cls_preds[0]
        # labels_3d = labels_3d[0]

        bbox_results = self._bbox_forward(pts_xyz, pts_feats, pts_batch_inds, rois)

        bbox_list = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['valid_roi_mask'],
            labels_3d,
            cls_preds,
            img_metas,
            cfg=self.test_cfg)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _bbox_forward_train(self, pts_xyz, pts_feats, batch_idx, sampling_results):

        rois = bbox3d2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(pts_xyz, pts_feats, batch_idx, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['valid_roi_mask'],
            rois,
            *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, pts_xyz, pts_feats, batch_idx, rois):

        assert pts_xyz.size(0) == pts_feats.size(0) == batch_idx.size(0)

        ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
            pts_xyz[:, :3], # intensity might be in pts_xyz
            batch_idx,
            rois[:, :8],
        )

        new_pts_feats = pts_feats[ext_pts_inds]
        new_pts_xyz = pts_xyz[ext_pts_inds]

        # def forward(self, pts_xyz, pts_features, pts_info, roi_inds, rois):

        cls_score, bbox_pred, valid_roi_mask = self.bbox_head(
            new_pts_xyz,
            new_pts_feats,
            ext_pts_info,
            ext_pts_roi_inds,
            rois,
        )

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            valid_roi_mask=valid_roi_mask,
        )

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
        assert len(proposal_list) == len(gt_bboxes_3d)
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_boxes, cur_scores, cur_pd_labels = proposal_list[batch_idx]
            # fake a box if no real proposal
            no_proposal = len(cur_boxes) == 0
            if no_proposal:
                # print('*******fake a box*******')
                cur_boxes = LiDARInstance3DBoxes(torch.tensor([[0,0,5,1,1,1,0]], dtype=torch.float32, device=cur_boxes.device))
                cur_scores = torch.tensor([0.0], dtype=torch.float32, device=cur_boxes.device)
                cur_pd_labels = torch.tensor([0], dtype=torch.int64, device=cur_boxes.device)

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
                    gt_cls_mask = (cur_gt_labels == i)
                    pred_cls_mask = (cur_pd_labels == i)
                    cur_assign_res = assigner.assign(
                        cur_boxes.tensor[pred_cls_mask, :7],
                        cur_gt_bboxes.tensor[gt_cls_mask, :7],
                        gt_labels=cur_gt_labels[gt_cls_mask])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_cls_mask.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    # gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_cls_mask] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] # - 1
                    batch_max_overlaps[
                        pred_cls_mask] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_cls_mask] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_boxes.tensor[:, :7],
                    cur_gt_bboxes.tensor[:, :7],
                    gt_labels=cur_gt_labels)
            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList, batch_data_samples: SampleList):
        raise NotImplementedError
