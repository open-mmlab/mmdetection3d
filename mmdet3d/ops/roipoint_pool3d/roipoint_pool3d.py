from torch import nn as nn
from torch.autograd import Function

from . import roipoint_pool3d_ext


class RoIPointPool3d(nn.Module):

    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d,
                                            self.pool_extra_width,
                                            self.num_sampled_points)


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

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
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
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
