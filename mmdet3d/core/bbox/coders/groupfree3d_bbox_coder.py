import numpy as np

from mmdet.core.bbox.builder import BBOX_CODERS
from .partial_bin_based_bbox_coder import PartialBinBasedBBoxCoder


@BBOX_CODERS.register_module()
class GroupFree3DBBoxCoder(PartialBinBasedBBoxCoder):
    """Partial bin based bbox coder for GroupFree3D.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        num_sizes (int): Number of size clusters.
        mean_sizes (list[list[int]]): Mean size of bboxes in each class.
        with_rot (bool): Whether the bbox is with rotation.
    """

    def __init__(self, num_dir_bins, num_sizes, mean_sizes, with_rot=True):
        super(GroupFree3DBBoxCoder, self).__init__(
            num_dir_bins=num_dir_bins,
            num_sizes=num_sizes,
            mean_sizes=mean_sizes,
            with_rot=with_rot)

    def split_pred(self, cls_preds, reg_preds, base_xyz, suffix=''):
        """Split predicted features to specific parts.

        Args:
            cls_preds (torch.Tensor): Class predicted features to split.
            reg_preds (torch.Tensor): Regression predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.
            suffix (str): Decode predictions with specific suffix.

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
        results['center_residual' + suffix] = \
            reg_preds_trans[..., start:end].contiguous()
        results['center' + suffix] = base_xyz + \
            reg_preds_trans[..., start:end].contiguous()
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class' + suffix] = \
            reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end].contiguous()
        start = end

        results['dir_res_norm' + suffix] = dir_res_norm
        results['dir_res' + suffix] = dir_res_norm * (
            np.pi / self.num_dir_bins)

        # decode size
        end += self.num_sizes
        results['size_class' +
                suffix] = reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_sizes * 3
        size_res_norm = reg_preds_trans[..., start:end]
        batch_size, num_proposal = reg_preds_trans.shape[:2]
        size_res_norm = size_res_norm.view(
            [batch_size, num_proposal, self.num_sizes, 3])
        start = end

        results['size_res_norm' + suffix] = size_res_norm.contiguous()
        mean_sizes = reg_preds.new_tensor(self.mean_sizes)
        results['size_res' + suffix] = (
            size_res_norm * mean_sizes.unsqueeze(0).unsqueeze(0))

        # decode objectness score
        start = 0
        # Group-Free-3D objectness output shape (B, N, 1)
        end = 1
        results['obj_scores' +
                suffix] = cls_preds_trans[..., start:end].contiguous()
        start = end

        # decode semantic score
        results['sem_scores' + suffix] = cls_preds_trans[...,
                                                         start:].contiguous()

        return results
