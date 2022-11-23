# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses.utils import weight_reduce_loss
from torch import nn as nn

from mmdet3d.registry import MODELS
from ..layers import PAConv, PAConvCUDA


def weight_correlation(conv):
    """Calculate correlations between kernel weights in Conv's weight bank as
    regularization loss. The cosine similarity is used as metrics.

    Args:
        conv (nn.Module): A Conv modules to be regularized.
            Currently we only support `PAConv` and `PAConvCUDA`.

    Returns:
        torch.Tensor: Correlations between each kernel weights in weight bank.
    """
    assert isinstance(conv, (PAConv, PAConvCUDA)), \
        f'unsupported module type {type(conv)}'
    kernels = conv.weight_bank  # [C_in, num_kernels * C_out]
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    num_kernels = conv.num_kernels

    # [num_kernels, Cin * Cout]
    flatten_kernels = kernels.view(in_channels, num_kernels, out_channels).\
        permute(1, 0, 2).reshape(num_kernels, -1)
    # [num_kernels, num_kernels]
    inner_product = torch.matmul(flatten_kernels, flatten_kernels.T)
    # [num_kernels, 1]
    kernel_norms = torch.sum(flatten_kernels**2, dim=-1, keepdim=True)**0.5
    # [num_kernels, num_kernels]
    kernel_norms = torch.matmul(kernel_norms, kernel_norms.T)
    cosine_sims = inner_product / kernel_norms
    # take upper triangular part excluding diagonal since we only compute
    # correlation between different kernels once
    # the square is to ensure positive loss, refer to:
    # https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg/tool/train.py#L208
    corr = torch.sum(torch.triu(cosine_sims, diagonal=1)**2)

    return corr


def paconv_regularization_loss(modules, reduction):
    """Computes correlation loss of PAConv weight kernels as regularization.

    Args:
        modules (List[nn.Module] | :obj:`generator`):
            A list or a python generator of torch.nn.Modules.
        reduction (str): Method to reduce losses among PAConv modules.
            The valid reduction method are none, sum or mean.

    Returns:
        torch.Tensor: Correlation loss of kernel weights.
    """
    corr_loss = []
    for module in modules:
        if isinstance(module, (PAConv, PAConvCUDA)):
            corr_loss.append(weight_correlation(module))
    corr_loss = torch.stack(corr_loss)

    # perform reduction
    corr_loss = weight_reduce_loss(corr_loss, reduction=reduction)

    return corr_loss


@MODELS.register_module()
class PAConvRegularizationLoss(nn.Module):
    """Calculate correlation loss of kernel weights in PAConv's weight bank.

    This is used as a regularization term in PAConv model training.

    Args:
        reduction (str): Method to reduce losses. The reduction is performed
            among all PAConv modules instead of prediction tensors.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PAConvRegularizationLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, modules, reduction_override=None, **kwargs):
        """Forward function of loss calculation.

        Args:
            modules (List[nn.Module] | :obj:`generator`):
                A list or a python generator of torch.nn.Modules.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: Correlation loss of kernel weights.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        return self.loss_weight * paconv_regularization_loss(
            modules, reduction=reduction)
