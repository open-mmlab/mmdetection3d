# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.autograd import Function

from . import interpolate_ext


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """Performs weighted linear interpolation on 3 features.

        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated from
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation

        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        interpolate_ext.three_interpolate_wrapper(B, c, m, n, features,
                                                  indices, weight, output)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of three interpolate.

        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.cuda.FloatTensor(B, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()

        interpolate_ext.three_interpolate_grad_wrapper(B, c, n, m,
                                                       grad_out_data, idx,
                                                       weight,
                                                       grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply
