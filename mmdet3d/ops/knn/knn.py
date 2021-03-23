import torch
from torch.autograd import Function

from . import knn_ext


class KNN(Function):
    """KNN (CUDA).

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
                defaults to False. Should not expicitly use this keyword
                when calling knn (=KNN.apply), just add the fourth param.

        Returns:
            Tensor: (B, k, npoint) tensor with the indicies of
                the features that form k-nearest neighbours.
        """
        assert k > 0

        if not transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()

        B, _, npoint = center_xyz.shape
        N = xyz.shape[2]

        assert center_xyz.is_contiguous()
        assert xyz.is_contiguous()

        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), \
            'center_xyz and xyz should be put on the same device'
        if torch.cuda.current_device() != center_xyz_device:
            torch.cuda.set_device(center_xyz_device)

        idx = center_xyz.new_zeros((B, k, npoint)).long()

        for bi in range(B):
            knn_ext.knn_wrapper(xyz[bi], N, center_xyz[bi], npoint, idx[bi], k)

        ctx.mark_non_differentiable(idx)

        idx -= 1

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None


knn = KNN.apply
