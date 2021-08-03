import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from .partial_bin_based_bbox_coder import PartialBinBasedBBoxCoder


@BBOX_CODERS.register_module()
class GroupFree3DBBoxCoder(PartialBinBasedBBoxCoder):
    """Modified partial bin based bbox coder for GroupFree3D.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        num_sizes (int): Number of size clusters.
        mean_sizes (list[list[int]]): Mean size of bboxes in each class.
        with_rot (bool): Whether the bbox is with rotation. Defaults to True.
        size_cls_agnostic (bool): Whether the predicted size is class-agnostic.
            Defaults to True.
    """

    def __init__(self,
                 num_dir_bins,
                 num_sizes,
                 mean_sizes,
                 with_rot=True,
                 size_cls_agnostic=True):
        super(GroupFree3DBBoxCoder, self).__init__(
            num_dir_bins=num_dir_bins,
            num_sizes=num_sizes,
            mean_sizes=mean_sizes,
            with_rot=with_rot)
        self.size_cls_agnostic = size_cls_agnostic

    def encode(self, gt_bboxes_3d, gt_labels_3d):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes \
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        # generate bbox size target
        size_target = gt_bboxes_3d.dims
        size_class_target = gt_labels_3d
        size_res_target = gt_bboxes_3d.dims - gt_bboxes_3d.tensor.new_tensor(
            self.mean_sizes)[size_class_target]

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d.yaw)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        return (center_target, size_target, size_class_target, size_res_target,
                dir_class_target, dir_res_target)

    def decode(self, bbox_out, prefix=''):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - center: predicted bottom center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size_class: predicted bbox size class.
                - size_res: predicted bbox size residual.
                - size: predicted class-agnostic bbox size
            prefix (str): Decode predictions with specific prefix.
                Defaults to ''.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (batch, n, 7).
        """
        center = bbox_out[f'{prefix}center']
        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out[f'{prefix}dir_class'], -1)
            dir_res = torch.gather(bbox_out[f'{prefix}dir_res'], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        # decode bbox size
        if self.size_cls_agnostic:
            bbox_size = bbox_out[f'{prefix}size'].reshape(
                batch_size, num_proposal, 3)
        else:
            size_class = torch.argmax(
                bbox_out[f'{prefix}size_class'], -1, keepdim=True)
            size_res = torch.gather(
                bbox_out[f'{prefix}size_res'], 2,
                size_class.unsqueeze(-1).repeat(1, 1, 1, 3))
            mean_sizes = center.new_tensor(self.mean_sizes)
            size_base = torch.index_select(mean_sizes, 0,
                                           size_class.reshape(-1))
            bbox_size = size_base.reshape(batch_size, num_proposal,
                                          -1) + size_res.squeeze(2)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def split_pred(self, cls_preds, reg_preds, base_xyz, prefix=''):
        """Split predicted features to specific parts.

        Args:
            cls_preds (torch.Tensor): Class predicted features to split.
            reg_preds (torch.Tensor): Regression predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.
            prefix (str): Decode predictions with specific prefix.
                Defaults to ''.

        Returns:
            dict[str, torch.Tensor]: Split results.
        """
        results = {}
        start, end = 0, 0

        cls_preds_trans = cls_preds.transpose(2, 1)
        reg_preds_trans = reg_preds.transpose(2, 1)

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results[f'{prefix}center_residual'] = \
            reg_preds_trans[..., start:end].contiguous()
        results[f'{prefix}center'] = base_xyz + \
            reg_preds_trans[..., start:end].contiguous()
        start = end

        # decode direction
        end += self.num_dir_bins
        results[f'{prefix}dir_class'] = \
            reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end].contiguous()
        start = end

        results[f'{prefix}dir_res_norm'] = dir_res_norm
        results[f'{prefix}dir_res'] = dir_res_norm * (
            np.pi / self.num_dir_bins)

        # decode size
        if self.size_cls_agnostic:
            end += 3
            results[f'{prefix}size'] = \
                reg_preds_trans[..., start:end].contiguous()
        else:
            end += self.num_sizes
            results[f'{prefix}size_class'] = reg_preds_trans[
                ..., start:end].contiguous()
            start = end

            end += self.num_sizes * 3
            size_res_norm = reg_preds_trans[..., start:end]
            batch_size, num_proposal = reg_preds_trans.shape[:2]
            size_res_norm = size_res_norm.view(
                [batch_size, num_proposal, self.num_sizes, 3])
            start = end

            results[f'{prefix}size_res_norm'] = size_res_norm.contiguous()
            mean_sizes = reg_preds.new_tensor(self.mean_sizes)
            results[f'{prefix}size_res'] = (
                size_res_norm * mean_sizes.unsqueeze(0).unsqueeze(0))

        # decode objectness score
        # Group-Free-3D objectness output shape (batch, proposal, 1)
        results[f'{prefix}obj_scores'] = cls_preds_trans[..., :1].contiguous()

        # decode semantic score
        results[f'{prefix}sem_scores'] = cls_preds_trans[..., 1:].contiguous()

        return results
