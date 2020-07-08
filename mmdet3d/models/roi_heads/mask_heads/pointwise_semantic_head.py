import torch
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import build_loss
from mmdet.core import multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class PointwiseSemanticHead(nn.Module):
    """Semantic segmentation head for point-wise segmentation.

    Predict point-wise segmentation and part regression results for PartA2.
    See `paper <https://arxiv.org/abs/1907.03670>`_ for more detials.

    Args:
        in_channels (int): The number of input channel.
        num_classes (int): The number of class.
        extra_width (float): Boxes enlarge width.
        loss_seg (dict): Config of segmentation loss.
        loss_part (dict): Config of part prediction loss.
    """

    def __init__(self,
                 in_channels,
                 num_classes=3,
                 extra_width=0.2,
                 seg_score_thr=0.3,
                 loss_seg=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_part=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super(PointwiseSemanticHead, self).__init__()
        self.extra_width = extra_width
        self.num_classes = num_classes
        self.seg_score_thr = seg_score_thr
        self.seg_cls_layer = nn.Linear(in_channels, 1, bias=True)
        self.seg_reg_layer = nn.Linear(in_channels, 3, bias=True)

        self.loss_seg = build_loss(loss_seg)
        self.loss_part = build_loss(loss_part)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Features from the first stage.

        Returns:
            dict: Part features, segmentation and part predictions.

                - seg_preds (torch.Tensor): Segment predictions.
                - part_preds (torch.Tensor): Part predictions.
                - part_feats (torch.Tensor): Feature predictions.
        """
        seg_preds = self.seg_cls_layer(x)  # (N, 1)
        part_preds = self.seg_reg_layer(x)  # (N, 3)

        seg_scores = torch.sigmoid(seg_preds).detach()
        seg_mask = (seg_scores > self.seg_score_thr)

        part_offsets = torch.sigmoid(part_preds).clone().detach()
        part_offsets[seg_mask.view(-1) == 0] = 0
        part_feats = torch.cat((part_offsets, seg_scores),
                               dim=-1)  # shape (npoints, 4)
        return dict(
            seg_preds=seg_preds, part_preds=part_preds, part_feats=part_feats)

    def get_targets_single(self, voxel_centers, gt_bboxes_3d, gt_labels_3d):
        """generate segmentation and part prediction targets for a single
        sample.

        Args:
            voxel_centers (torch.Tensor): The center of voxels in shape \
                (voxel_num, 3).
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in \
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in \
                shape (box_num).

        Returns:
            tuple[torch.Tensor]: Segmentation targets with shape [voxel_num] \
                part prediction targets with shape [voxel_num, 3]
        """
        gt_bboxes_3d = gt_bboxes_3d.to(voxel_centers.device)
        enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

        part_targets = voxel_centers.new_zeros((voxel_centers.shape[0], 3),
                                               dtype=torch.float32)
        box_idx = gt_bboxes_3d.points_in_boxes(voxel_centers)
        enlarge_box_idx = enlarged_gt_boxes.points_in_boxes(
            voxel_centers).long()

        gt_labels_pad = F.pad(
            gt_labels_3d, (1, 0), mode='constant', value=self.num_classes)
        seg_targets = gt_labels_pad[(box_idx.long() + 1)]
        fg_pt_flag = box_idx > -1
        ignore_flag = fg_pt_flag ^ (enlarge_box_idx > -1)
        seg_targets[ignore_flag] = -1

        for k in range(len(gt_bboxes_3d)):
            k_box_flag = box_idx == k
            # no point in current box (caused by velodyne reduce)
            if not k_box_flag.any():
                continue
            fg_voxels = voxel_centers[k_box_flag]
            transformed_voxels = fg_voxels - gt_bboxes_3d.bottom_center[k]
            transformed_voxels = rotation_3d_in_axis(
                transformed_voxels.unsqueeze(0),
                -gt_bboxes_3d.yaw[k].view(1),
                axis=2)
            part_targets[k_box_flag] = transformed_voxels / gt_bboxes_3d.dims[
                k] + voxel_centers.new_tensor([0.5, 0.5, 0])

        part_targets = torch.clamp(part_targets, min=0)
        return seg_targets, part_targets

    def get_targets(self, voxels_dict, gt_bboxes_3d, gt_labels_3d):
        """generate segmentation and part prediction targets.

        Args:
            voxel_centers (torch.Tensor): The center of voxels in shape \
                (voxel_num, 3).
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in \
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in \
                shape (box_num).

        Returns:
            dict: Prediction targets

                - seg_targets (torch.Tensor): Segmentation targets \
                    with shape [voxel_num].
                - part_targets (torch.Tensor): Part prediction targets \
                    with shape [voxel_num, 3].
        """
        batch_size = len(gt_labels_3d)
        voxel_center_list = []
        for idx in range(batch_size):
            coords_idx = voxels_dict['coors'][:, 0] == idx
            voxel_center_list.append(voxels_dict['voxel_centers'][coords_idx])

        seg_targets, part_targets = multi_apply(self.get_targets_single,
                                                voxel_center_list,
                                                gt_bboxes_3d, gt_labels_3d)
        seg_targets = torch.cat(seg_targets, dim=0)
        part_targets = torch.cat(part_targets, dim=0)
        return dict(seg_targets=seg_targets, part_targets=part_targets)

    def loss(self, semantic_results, semantic_targets):
        """Calculate point-wise segmentation and part prediction losses.

        Args:
            semantic_results (dict): Results from semantic head.

                - seg_preds: Segmentation predictions.
                - part_preds: Part predictions.

            semantic_targets (dict): Targets of semantic results.

                - seg_preds: Segmentation targets.
                - part_preds: Part targets.

        Returns:
            dict: Loss of segmentation and part prediction.

                - loss_seg (torch.Tensor): Segmentation prediction loss.
                - loss_part (torch.Tensor): Part prediction loss.
        """
        seg_preds = semantic_results['seg_preds']
        part_preds = semantic_results['part_preds']
        seg_targets = semantic_targets['seg_targets']
        part_targets = semantic_targets['part_targets']

        pos_mask = (seg_targets > -1) & (seg_targets < self.num_classes)
        binary_seg_target = pos_mask.long()
        pos = pos_mask.float()
        neg = (seg_targets == self.num_classes).float()
        seg_weights = pos + neg
        pos_normalizer = pos.sum()
        seg_weights = seg_weights / torch.clamp(pos_normalizer, min=1.0)
        loss_seg = self.loss_seg(seg_preds, binary_seg_target, seg_weights)

        if pos_normalizer > 0:
            loss_part = self.loss_part(part_preds[pos_mask],
                                       part_targets[pos_mask])
        else:
            # fake a part loss
            loss_part = loss_seg.new_tensor(0)

        return dict(loss_seg=loss_seg, loss_part=loss_part)
