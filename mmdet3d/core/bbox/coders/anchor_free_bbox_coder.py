# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from .partial_bin_based_bbox_coder import PartialBinBasedBBoxCoder


@BBOX_CODERS.register_module()
class AnchorFreeBBoxCoder(PartialBinBasedBBoxCoder):
    """Anchor free bbox coder for 3D boxes.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        with_rot (bool): Whether the bbox is with rotation.
    """

    def __init__(self, num_dir_bins, with_rot=True):
        super(AnchorFreeBBoxCoder, self).__init__(
            num_dir_bins, 0, [], with_rot=with_rot)
        self.num_dir_bins = num_dir_bins
        self.with_rot = with_rot

    def encode(self, gt_bboxes_3d, gt_labels_3d):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        # generate bbox size target
        size_res_target = gt_bboxes_3d.dims / 2

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d.yaw)
            dir_res_target /= (2 * np.pi / self.num_dir_bins)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        return (center_target, size_res_target, dir_class_target,
                dir_res_target)

    def decode(self, bbox_out):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - center: predicted bottom center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size: predicted bbox size.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (batch, n, 7).
        """
        center = bbox_out['center']
        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out['dir_class'], -1)
            dir_res = torch.gather(bbox_out['dir_res'], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        # decode bbox size
        bbox_size = torch.clamp(bbox_out['size'] * 2, min=0.1)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def split_pred(self, cls_preds, reg_preds, base_xyz):
        """Split predicted features to specific parts.

        Args:
            cls_preds (torch.Tensor): Class predicted features to split.
            reg_preds (torch.Tensor): Regression predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.

        Returns:
            dict[str, torch.Tensor]: Split results.
        """
        results = {}
        results['obj_scores'] = cls_preds

        start, end = 0, 0
        reg_preds_trans = reg_preds.transpose(2, 1)

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['center_offset'] = reg_preds_trans[..., start:end]
        results['center'] = base_xyz.detach() + reg_preds_trans[..., start:end]
        start = end

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['size'] = reg_preds_trans[..., start:end]
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class'] = reg_preds_trans[..., start:end]
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end]
        start = end

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (2 * np.pi / self.num_dir_bins)

        return results
