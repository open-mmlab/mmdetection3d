import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class PartialBinBasedBBoxCoder(BaseBBoxCoder):
    """Partial bin based bbox coder

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        num_sizes (int): Number of size clusters.
        mean_sizes (list[list[int]]): Mean size of bboxes in each class.
        with_rot (bool): Whether the bbox is with rotation.
    """

    def __init__(self, num_dir_bins, num_sizes, mean_sizes, with_rot=True):
        super(PartialBinBasedBBoxCoder, self).__init__()
        assert len(mean_sizes) == num_sizes
        self.num_dir_bins = num_dir_bins
        self.num_sizes = num_sizes
        self.mean_sizes = mean_sizes
        self.with_rot = with_rot

    def encode(self, gt_bboxes_3d, gt_labels_3d):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): gt bboxes with shape (n, 7).
            gt_labels_3d (torch.Tensor): gt classes.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        # generate bbox size target
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

        return (center_target, size_class_target, size_res_target,
                dir_class_target, dir_res_target)

    def decode(self, bbox_out):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): predictions from model, should contain keys below
                - center: predicted bottom center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size_class: predicted bbox size class.
                - size_res: predicted bbox size residual.

        Returns:
            torch.Tensor: decoded bbox3d with shape (batch, n, 7)
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
        size_class = torch.argmax(bbox_out['size_class'], -1, keepdim=True)
        size_res = torch.gather(bbox_out['size_res'], 2,
                                size_class.unsqueeze(-1).repeat(1, 1, 1, 3))
        mean_sizes = center.new_tensor(self.mean_sizes)
        size_base = torch.index_select(mean_sizes, 0, size_class.reshape(-1))
        bbox_size = size_base.reshape(batch_size, num_proposal,
                                      -1) + size_res.squeeze(2)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def split_pred(self, preds, base_xyz):
        """Split predicted features to specific parts.

        Args:
            preds (torch.Tensor): predicted features to split.
            base_xyz (torch.Tensor): coordinates of points.

        Returns:
            dict: split results.
        """
        results = {}
        start, end = 0, 0
        preds_trans = preds.transpose(2, 1)

        # decode objectness score
        end += 2
        results['obj_scores'] = preds_trans[..., start:end]
        start = end

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['center'] = base_xyz + preds_trans[..., start:end]
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class'] = preds_trans[..., start:end]
        start = end

        end += self.num_dir_bins
        dir_res_norm = preds_trans[..., start:end]
        start = end

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (np.pi / self.num_dir_bins)

        # decode size
        end += self.num_sizes
        results['size_class'] = preds_trans[..., start:end]
        start = end

        end += self.num_sizes * 3
        size_res_norm = preds_trans[..., start:end]
        batch_size, num_proposal = preds_trans.shape[:2]
        size_res_norm = size_res_norm.view(
            [batch_size, num_proposal, self.num_sizes, 3])
        start = end

        results['size_res_norm'] = size_res_norm
        mean_sizes = preds.new_tensor(self.mean_sizes)
        results['size_res'] = (
            size_res_norm * mean_sizes.unsqueeze(0).unsqueeze(0))

        # decode semantic score
        results['sem_scores'] = preds_trans[..., start:]

        return results

    def angle2class(self, angle):
        """Convert continuous angle to a discrete class and a residual.

        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.

        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).

        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (
            angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """Inverse function to angle2class

        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].

        Returns:
            torch.Tensor: angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle
