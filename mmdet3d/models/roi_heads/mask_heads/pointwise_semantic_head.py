import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.core import multi_apply
from mmdet3d.core.bbox import box_torch_ops
from mmdet3d.models.builder import build_loss
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet.models import HEADS


@HEADS.register_module()
class PointwiseSemanticHead(nn.Module):
    """Semantic segmentation head for point-wise segmentation.

    Predict point-wise segmentation and part regression results for PartA2.
    See https://arxiv.org/abs/1907.03670 for more detials.

    Args:
        in_channels (int): the number of input channel.
        num_classes (int): the number of class.
        extra_width (float): boxes enlarge width.
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
        """generate segmentation and part prediction targets

        Args:
            voxel_centers (torch.Tensor): shape [voxel_num, 3],
                the center of voxels
            gt_bboxes_3d (torch.Tensor): shape [box_num, 7], gt boxes
            gt_labels_3d (torch.Tensor): shape [box_num], class label of gt

        Returns:
            tuple : segmentation targets with shape [voxel_num]
                part prediction targets with shape [voxel_num, 3]
        """
        enlarged_gt_boxes = box_torch_ops.enlarge_box3d_lidar(
            gt_bboxes_3d, extra_width=self.extra_width)
        part_targets = voxel_centers.new_zeros((voxel_centers.shape[0], 3),
                                               dtype=torch.float32)
        box_idx = points_in_boxes_gpu(
            voxel_centers.unsqueeze(0),
            gt_bboxes_3d.unsqueeze(0)).squeeze(0)  # -1 ~ box_num
        enlarge_box_idx = points_in_boxes_gpu(
            voxel_centers.unsqueeze(0),
            enlarged_gt_boxes.unsqueeze(0)).squeeze(0).long()  # -1 ~ box_num

        gt_labels_pad = F.pad(
            gt_labels_3d, (1, 0), mode='constant', value=self.num_classes)
        seg_targets = gt_labels_pad[(box_idx.long() + 1)]
        fg_pt_flag = box_idx > -1
        ignore_flag = fg_pt_flag ^ (enlarge_box_idx > -1)
        seg_targets[ignore_flag] = -1

        for k in range(gt_bboxes_3d.shape[0]):
            k_box_flag = box_idx == k
            # no point in current box (caused by velodyne reduce)
            if not k_box_flag.any():
                continue
            fg_voxels = voxel_centers[k_box_flag]
            transformed_voxels = fg_voxels - gt_bboxes_3d[k, 0:3]
            transformed_voxels = box_torch_ops.rotation_3d_in_axis(
                transformed_voxels.unsqueeze(0),
                -gt_bboxes_3d[k, 6].view(1),
                axis=2)
            part_targets[k_box_flag] = transformed_voxels / gt_bboxes_3d[
                k, 3:6] + voxel_centers.new_tensor([0.5, 0.5, 0])

        part_targets = torch.clamp(part_targets, min=0)
        return seg_targets, part_targets

    def get_targets(self, voxels_dict, gt_bboxes_3d, gt_labels_3d):
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
            semantic_targets (dict): Targets of semantic results.

        Returns:
            dict: loss of segmentation and part prediction.
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
