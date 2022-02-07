# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.autograd import Function

from . import knn_ext


class KNN(Function):
    r"""KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx,
                k: int,
                xyz: torch.Tensor,
                center_xyz: torch.Tensor = None,
                transposed: bool = False) -> torch.Tensor:
        """Forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) if transposed == False,
                else (B, 3, npoint). centers of the knn query.
            transposed (bool): whether the input tensors are transposed.
                defaults to False. Should not explicitly use this keyword
                when calling knn (=KNN.apply), just add the fourth param.

        Returns:
            Tensor: (B, k, npoint) tensor with the indices of
                the features that form k-nearest neighbours.
        """
        assert k > 0

        if center_xyz is None:
            center_xyz = xyz

        if transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()

        assert xyz.is_contiguous()  # [B, N, 3]
        assert center_xyz.is_contiguous()  # [B, npoint, 3]

        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), \
            'center_xyz and xyz should be put on the same device'
        if torch.cuda.current_device() != center_xyz_device:
            torch.cuda.set_device(center_xyz_device)

        B, npoint, _ = center_xyz.shape
        N = xyz.shape[1]

        idx = center_xyz.new_zeros((B, npoint, k)).int()
        dist2 = center_xyz.new_zeros((B, npoint, k)).float()

        knn_ext.knn_wrapper(B, N, npoint, k, xyz, center_xyz, idx, dist2)
        # idx shape to [B, k, npoint]
        idx = idx.transpose(2, 1).contiguous()
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knn = KNN.apply
