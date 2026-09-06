# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import torch
# --- Assigner base classes ---
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner

from mmdet3d.registry import TASK_UTILS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


# ---------------------------------------------------------------------------
# Match cost functions (previously registered to MATCH_COST)
# ---------------------------------------------------------------------------
@TASK_UTILS.register_module(force=True)
class FocalLossCost(object):
    """Focal loss cost for classification (raw tensor API).

    Replaces mmdet.FocalLossCost which expects InstanceData objects.
    """

    def __init__(self,
                 weight=1.0,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def __call__(self, cls_pred, gt_labels):
        """Compute focal loss cost.

        Args:
            cls_pred (Tensor): (num_bboxes, num_classes) classification scores
                in sigmoid.
            gt_labels (Tensor): (num_gts,) ground truth labels.

        Returns:
            Tensor: (num_bboxes, num_gts) cost matrix.
        """
        cls_pred = cls_pred.float()
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@TASK_UTILS.register_module()
class BBox3DL1Cost(object):
    """L1 cost for 3D bounding box regression."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg):
        reg_cost = torch.cdist(bboxes, gt_bboxes, p=1)
        return reg_cost * self.weight


@TASK_UTILS.register_module()
class BBoxBEVL1Cost(object):
    """L1 cost for BEV center regression (normalized to [0, 1])."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg):
        pc_start = bboxes.new(train_cfg['point_cloud_range'][0:2])
        pc_range = (
            bboxes.new(train_cfg['point_cloud_range'][3:5]) -
            bboxes.new(train_cfg['point_cloud_range'][0:2]))
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(
            normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@TASK_UTILS.register_module()
class IoU3DCost(object):
    """IoU cost for 3D bounding boxes."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        iou_cost = -iou
        return iou_cost * self.weight


# ---------------------------------------------------------------------------
# Assigners (previously registered to BBOX_ASSIGNERS)
# ---------------------------------------------------------------------------
@TASK_UTILS.register_module()
class HeuristicAssigner3D(BaseAssigner):
    """Heuristic assigner that matches each GT to its nearest prediction within
    a distance threshold."""

    def __init__(self,
                 dist_thre=100,
                 iou_calculator=dict(
                     type='BboxOverlaps3D', coordinate='lidar')):
        self.dist_thre = dist_thre
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               query_labels=None):
        dist_thre = self.dist_thre
        num_gts, num_bboxes = len(gt_bboxes), len(bboxes)

        # [num_gts, num_bboxes]
        bev_dist = torch.norm(
            bboxes[:, 0:2][None, :, :] - gt_bboxes[:, 0:2][:, None, :], dim=-1)
        if query_labels is not None:
            not_same_class = (query_labels[None] != gt_labels[:, None])
            bev_dist += not_same_class * dist_thre

        nearest_values, nearest_indices = bev_dist.min(1)
        assigned_gt_inds = torch.ones([num_bboxes]).to(bboxes) * 0
        assigned_gt_vals = torch.ones([num_bboxes]).to(bboxes) * 10000
        assigned_gt_labels = torch.ones([num_bboxes]).to(bboxes) * -1

        for idx_gts in range(num_gts):
            idx_pred = nearest_indices[idx_gts]
            if bev_dist[idx_gts, idx_pred] <= dist_thre:
                if bev_dist[idx_gts, idx_pred] < assigned_gt_vals[idx_pred]:
                    assigned_gt_vals[idx_pred] = bev_dist[idx_gts, idx_pred]
                    assigned_gt_inds[idx_pred] = idx_gts + 1
                    assigned_gt_labels[idx_pred] = gt_labels[idx_gts]

        max_overlaps = torch.zeros([num_bboxes]).to(bboxes)
        matched_indices = torch.where(assigned_gt_inds > 0)
        num_matched = matched_indices[0].numel()

        if num_matched > 0:
            matched_iou = self.iou_calculator(
                gt_bboxes[assigned_gt_inds[matched_indices].long() - 1],
                bboxes[matched_indices]).diag()
            max_overlaps[matched_indices] = matched_iou

        return AssignResult(
            num_gts,
            assigned_gt_inds.long(),
            max_overlaps,
            labels=assigned_gt_labels)


@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Hungarian assigner for 3D object detection.

    Uses a combination of classification cost, BEV L1 regression cost, and IoU
    cost to compute optimal assignment via the Hungarian algorithm.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxBEVL1Cost', weight=1.0),
                 iou_cost=dict(type='IoU3DCost', weight=1.0),
                 iou_calculator=dict(
                     type='BboxOverlaps3D', coordinate='lidar')):
        self.cls_cost = TASK_UTILS.build(cls_cost)

        self.reg_cost = TASK_UTILS.build(reg_cost)

        self.iou_cost = TASK_UTILS.build(iou_cost)

        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes, ),
                                           -1,
                                           dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes, ), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)

        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)

        iou = self.iou_calculator(bboxes, gt_bboxes)

        iou_cost = self.iou_cost(iou)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost_cpu = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_cpu)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        # 4. assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds,
                                             matched_col_inds]

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
