import torch
from torch.autograd import Function
import dynamic_point_pool_ext


class DynamicPointPoolFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, extra_wlh, max_inbox_point, max_all_pts=50000):
        """RoIAwarePool3d function forward.

        Args:
            rois (torch.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (int): 0 (max pool) or 1 (average pool)

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """

        # pts_inds, roi_inds, pts_norm_xyz, pts_offset = dynamic_point_pool_ext.forward(rois, pts)
        out_pts_idx = -1 * pts.new_ones(max_all_pts, dtype=torch.long)
        out_roi_idx = -1 * pts.new_ones(max_all_pts, dtype=torch.long)
        out_pts_feats = pts.new_zeros(max_all_pts, 13, dtype=torch.float)

        assert len(rois) > 0
        dynamic_point_pool_ext.forward(rois, pts, extra_wlh, max_inbox_point, out_pts_idx, out_roi_idx, out_pts_feats)
        # Because of cuda block layout, the out_roi_idx is automatically sorted, but not strictly guaranteed.
        valid_mask = out_pts_idx >= 0

        if not valid_mask.any():
            # fake a non-empty input
            out_pts_idx = out_pts_idx[0:1]
            out_roi_idx = out_roi_idx[0:1]
            out_pts_feats = out_pts_feats[0:1, :]
        else:
            out_pts_idx = out_pts_idx[valid_mask]
            out_roi_idx = out_roi_idx[valid_mask]
            out_pts_feats = out_pts_feats[valid_mask]
            unique_roi_idx = torch.unique(out_roi_idx)

        ctx.mark_non_differentiable(out_pts_idx)
        ctx.mark_non_differentiable(out_roi_idx)
        ctx.mark_non_differentiable(out_pts_feats)

        return out_pts_idx, out_roi_idx, out_pts_feats

    @staticmethod
    def backward(ctx, g1, g2, g3):

        return None, None, None, None, None


dynamic_point_pool = DynamicPointPoolFunction.apply
