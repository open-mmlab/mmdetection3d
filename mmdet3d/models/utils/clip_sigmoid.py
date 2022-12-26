# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor


def clip_sigmoid(x: Tensor, eps: float = 1e-4) -> Tensor:
    """Sigmoid function for input feature.

    Args:
        x (Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to.
            Defaults to 1e-4.

    Returns:
        Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y
