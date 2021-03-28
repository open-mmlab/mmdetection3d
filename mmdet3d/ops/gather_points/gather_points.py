import torch
from torch.autograd import Function

from . import gather_points_ext


class GatherPoints(Function):
    """Gather Points.

    Gather points with given index.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) features to gather.
            indices (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()

        B, npoint = indices.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        gather_points_ext.gather_points_wrapper(B, C, N, npoint, features,
                                                indices, output)

        ctx.for_backwards = (indices, C, N)
        ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        gather_points_ext.gather_points_grad_wrapper(B, C, N, npoint,
                                                     grad_out_data, idx,
                                                     grad_features.data)
        return grad_features, None


gather_points = GatherPoints.apply
