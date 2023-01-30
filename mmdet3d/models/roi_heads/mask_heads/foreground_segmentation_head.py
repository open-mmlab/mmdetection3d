# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from mmcv.cnn.bricks import build_norm_layer
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import InstanceList


@MODELS.register_module()
class ForegroundSegmentationHead(BaseModule):
    """Foreground segmentation head.

    Args:
        in_channels (int): The number of input channel.
        mlp_channels (tuple[int]): Specify of mlp channels. Defaults
            to (256, 256).
        extra_width (float): Boxes enlarge width. Default used 0.1.
        norm_cfg (dict): Type of normalization method. Defaults to
            dict(type='BN1d', eps=1e-5, momentum=0.1).
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        loss_seg (dict): Config of segmentation loss. Defaults to
            dict(type='mmdet.FocalLoss')
    """

    def __init__(
        self,
        in_channels: int,
        mlp_channels: Tuple[int] = (256, 256),
        extra_width: float = 0.1,
        norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
        init_cfg: Optional[dict] = None,
        loss_seg: dict = dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            activated=True,
            loss_weight=1.0)
    ) -> None:
        super(ForegroundSegmentationHead, self).__init__(init_cfg=init_cfg)
        self.extra_width = extra_width
        self.num_classes = 1

        self.in_channels = in_channels
        self.use_sigmoid_cls = loss_seg.get('use_sigmoid', False)

        out_channels = 1
        if self.use_sigmoid_cls:
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels + 1

        mlps_layers = []
        cin = in_channels
        for mlp in mlp_channels:
            mlps_layers.extend([
                nn.Linear(cin, mlp, bias=False),
                build_norm_layer(norm_cfg, mlp)[1],
                nn.ReLU()
            ])
            cin = mlp
        mlps_layers.append(nn.Linear(cin, self.out_channels, bias=True))

        self.seg_cls_layer = nn.Sequential(*mlps_layers)

        self.loss_seg = MODELS.build(loss_seg)

    def forward(self, feats: torch.Tensor) -> dict:
        """Forward head.

        Args:
            feats (torch.Tensor): Point-wise features.

        Returns:
            dict: Segment predictions.
        """
        seg_preds = self.seg_cls_layer(feats)
        return dict(seg_preds=seg_preds)

    def _get_targets_single(self, point_xyz: torch.Tensor,
                            gt_bboxes_3d: InstanceData,
                            gt_labels_3d: torch.Tensor) -> torch.Tensor:
        """generate segmentation targets for a single sample.

        Args:
            point_xyz (torch.Tensor): Coordinate of points.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in
                shape (box_num).

        Returns:
            torch.Tensor: Points class labels.
        """
        point_cls_labels_single = point_xyz.new_zeros(
            point_xyz.shape[0]).long()
        enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

        box_idxs_of_pts = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
        extend_box_idxs_of_pts = enlarged_gt_boxes.points_in_boxes_part(
            point_xyz).long()
        box_fg_flag = box_idxs_of_pts >= 0
        fg_flag = box_fg_flag.clone()
        ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
        point_cls_labels_single[ignore_flag] = -1
        gt_box_of_fg_points = gt_labels_3d[box_idxs_of_pts[fg_flag]]
        point_cls_labels_single[
            fg_flag] = 1 if self.num_classes == 1 else\
            gt_box_of_fg_points.long()
        return point_cls_labels_single,

    def get_targets(self, points_bxyz: torch.Tensor,
                    batch_gt_instances_3d: InstanceList) -> dict:
        """Generate segmentation targets.

        Args:
            points_bxyz (torch.Tensor): The coordinates of point in shape
                (B, num_points, 3).
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            dict: Prediction targets
                - seg_targets (torch.Tensor): Segmentation targets.
        """
        batch_size = len(batch_gt_instances_3d)
        points_xyz_list = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        for idx in range(batch_size):
            coords_idx = points_bxyz[:, 0] == idx
            points_xyz_list.append(points_bxyz[coords_idx][..., 1:])
            gt_bboxes_3d.append(batch_gt_instances_3d[idx].bboxes_3d)
            gt_labels_3d.append(batch_gt_instances_3d[idx].labels_3d)
        seg_targets, = multi_apply(self._get_targets_single, points_xyz_list,
                                   gt_bboxes_3d, gt_labels_3d)
        seg_targets = torch.cat(seg_targets, dim=0)
        return dict(seg_targets=seg_targets)

    def loss(self, semantic_results: dict,
             semantic_targets: dict) -> Dict[str, torch.Tensor]:
        """Calculate point-wise segmentation losses.

        Args:
            semantic_results (dict): Results from semantic head.
            semantic_targets (dict): Targets of semantic results.

        Returns:
            dict: Loss of segmentation.

            - loss_semantic (torch.Tensor): Segmentation prediction loss.
        """
        seg_preds = semantic_results['seg_preds']
        seg_targets = semantic_targets['seg_targets']

        positives = (seg_targets > 0)

        negative_cls_weights = (seg_targets == 0).float()
        seg_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        seg_weights /= torch.clamp(pos_normalizer, min=1.0)

        seg_preds = torch.sigmoid(seg_preds)
        loss_seg = self.loss_seg(seg_preds, (~positives).long(), seg_weights)
        return dict(loss_semantic=loss_seg)
