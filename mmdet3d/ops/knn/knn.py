import torch
from torch.autograd import Function

from . import knn_ext


class KNN(Function):
    """KNN.

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx,
                k: int,
                xyz: torch.Tensor,
                center_xyz: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        """forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) if transposed == False,
                else (B, 3, npoint). centers of the knn query.
            transposed (bool): whether the input tensors are transposed.
                defaults to False.

        Returns:
            Tensor: (B, k, npoint) tensor with the indicies of
                the features that form k-nearest neighbours.
        """
        assert k > 0

        B, npoint = center_xyz.shape[:2]
        N = xyz.shape[1]

        if not transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()

        assert center_xyz.is_contiguous()
        assert xyz.is_contiguous()

        idx = torch.cuda.LongTensor(B, k, npoint).zero_()

        for bi in range(B):
            knn_ext.knn_wrapper(xyz[bi], N, center_xyz[bi], npoint, idx[bi], k)

        ctx.mark_non_differentiable(idx)

        idx -= 1

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None


knn = KNN.apply
