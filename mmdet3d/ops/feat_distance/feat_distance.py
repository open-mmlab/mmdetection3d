import torch
from torch.autograd import Function

from . import feat_distance_ext


class FeatDistance(Function):
    """Feature distance.

    Calculate the distance between feature maps.
    """

    @staticmethod
    def forward(ctx, point_feat_a: torch.Tensor,
                point_feat_b: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            point_feat_a (Tensor): (B, N, C) Feature vector of each point.
            point_feat_b (Tensor): (B, M, C) Feature vector of each point.

        Returns:
            Tensor: (B, N, M) Distance between each pair points.
        """
        assert point_feat_a.is_contiguous()
        assert point_feat_b.is_contiguous()
        assert point_feat_a.shape[2] == point_feat_b.shape[2]

        B, N, C = point_feat_a.size()
        _, M, _ = point_feat_b.size()
        distance = torch.cuda.FloatTensor(B, N, M).zero_()

        feat_distance_ext.feat_distance_wrapper(B, N, M, C, point_feat_a,
                                                point_feat_b, distance)
        ctx.mark_non_differentiable(distance)
        return distance

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


calc_feat_distance = FeatDistance.apply
