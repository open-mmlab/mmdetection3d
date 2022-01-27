# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn as nn
from torch.autograd import Function

from . import roipoint_pool3d_ext


class RoIPointPool3d(nn.Module):

    def __init__(self, num_sampled_points=512):
        super().__init__()
        """
        Args:
            num_sampled_points (int): Number of samples in each roi
        """
        self.num_sampled_points = num_sampled_points

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points (torch.Tensor): Input points whose shape is BxNx3
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            torch.Tensor: (B, M, 512, 3 + C) pooled_features
            torch.Tensor: (B, M) pooled_empty_flag
        """
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d,
                                            self.num_sampled_points)


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(ctx, points, point_features, boxes3d, num_sampled_points=512):
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, 3)
            point_features (torch.Tensor): Input points features shape is \
                (B, N, C)
            boxes3d (torch.Tensor): Input bounding boxes whose shape is \
                (B, M, 7)
            num_sampled_points (int): the num of sampled points

        Returns:
            torch.Tensor: (B, M, 512, 3 + C) pooled_features
            torch.Tensor: (B, M) pooled_empty_flag
        """
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[
            1], point_features.shape[2]
        pooled_boxes3d = boxes3d.view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros(
            (batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros(
            (batch_size, boxes_num)).int()

        roipoint_pool3d_ext.forward(points.contiguous(),
                                    pooled_boxes3d.contiguous(),
                                    point_features.contiguous(),
                                    pooled_features, pooled_empty_flag)

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


if __name__ == '__main__':
    pass
