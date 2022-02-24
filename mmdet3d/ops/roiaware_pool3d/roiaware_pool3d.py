# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from torch import nn as nn
from torch.autograd import Function

from . import roiaware_pool3d_ext


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_per_voxel=128, mode='max'):
        super().__init__()
        """RoIAwarePool3d module

        Args:
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (str): 'max' or 'avg'
        """
        self.out_size = out_size
        self.max_pts_per_voxel = max_pts_per_voxel
        assert mode in ['max', 'avg']
        pool_method_map = {'max': 0, 'avg': 1}
        self.mode = pool_method_map[mode]

    def forward(self, rois, pts, pts_feature):
        """RoIAwarePool3d module forward.

        Args:
            rois (torch.Tensor): [N, 7],in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """

        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature,
                                            self.out_size,
                                            self.max_pts_per_voxel, self.mode)


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_per_voxel,
                mode):
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

        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            assert mmcv.is_tuple_of(out_size, int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros(
            (num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros(
            (num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros(
            (num_rois, out_x, out_y, out_z, max_pts_per_voxel),
            dtype=torch.int)

        roiaware_pool3d_ext.forward(rois, pts, pts_feature, argmax,
                                    pts_idx_of_voxels, pooled_features, mode)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, mode,
                                            num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """RoIAwarePool3d function forward.

        Args:
            grad_out (torch.Tensor): [N, out_x, out_y, out_z, C]
        Returns:
            grad_in (torch.Tensor): [npoints, C]
        """
        ret = ctx.roiaware_pool3d_for_backward
        pts_idx_of_voxels, argmax, mode, num_pts, num_channels = ret

        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_ext.backward(pts_idx_of_voxels, argmax,
                                     grad_out.contiguous(), grad_in, mode)

        return None, None, grad_in, None, None, None


if __name__ == '__main__':
    pass
