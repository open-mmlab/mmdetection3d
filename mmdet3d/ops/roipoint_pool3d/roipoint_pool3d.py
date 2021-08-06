from torch import nn as nn
from torch.autograd import Function

from . import roipoint_pool3d_ext


class RoIPointPool3d(nn.Module):

    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        """

        Args:
            num_sampled_points (int): Number of samples in each roi
            extra_width (float): Extra width to enlarge the box
        """
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

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
                                            self.pool_extra_width,
                                            self.num_sampled_points)


def enlarge_box3d(boxes3d, extra_width=0):
    """
    Args:
        boxes3d(torch.Tensor): Float matrix of N x box_dim, \
            Each row is (x, y, z, dx, dy, dz, yaw), and (x, y, z) is \
            bottom_center
        extra_width (float): Extra width to enlarge the box.

    Returns:
        torch.Tensor: (B, M, 7) the enlarged bounding boxes
    """
    large_boxes3d = boxes3d.clone()

    large_boxes3d[..., 3:6] += boxes3d.new_tensor(extra_width)
    large_boxes3d[..., 2] -= boxes3d.new_tensor(extra_width / 2)
    return large_boxes3d


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(ctx,
                points,
                point_features,
                boxes3d,
                pool_extra_width,
                num_sampled_points=512):
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, 3)
            point_features (torch.Tensor): Input points features shape is \
                (B, N, C)
            boxes3d (torch.Tensor): Input bounding boxes whose shape is \
                (B, M, 7)
            pool_extra_width (float): Extra width to enlarge the box
            num_sampled_points (int): the num of sampled points

        Returns:
            torch.Tensor: (B, M, 512, 3 + C) pooled_features
            torch.Tensor: (B, M) pooled_empty_flag
        """
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[
            1], point_features.shape[2]
        pooled_boxes3d = enlarge_box3d(boxes3d.view(-1, 7),
                                       pool_extra_width).view(
                                           batch_size, -1, 7)

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
