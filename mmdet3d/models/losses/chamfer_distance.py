# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet3d.registry import MODELS


def chamfer_distance(
        src: Tensor,
        dst: Tensor,
        src_weight: Union[Tensor, float] = 1.0,
        dst_weight: Union[Tensor, float] = 1.0,
        criterion_mode: str = 'l2',
        reduction: str = 'mean') -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate Chamfer Distance of two sets.

    Args:
        src (Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (Tensor or float): Weight of source loss. Defaults to 1.0.
        dst_weight (Tensor or float): Weight of destination loss.
            Defaults to 1.0.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are 'smooth_l1', 'l1' or 'l2'. Defaults to 'l2'.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
            Defaults to 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (Tensor): The min distance
              from source to destination.
            - loss_dst (Tensor): The min distance
              from destination to source.
            - indices1 (Tensor): Index the min distance point
              for each point in source to destination.
            - indices2 (Tensor): Index the min distance point
              for each point in destination to source.
    """

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2


@MODELS.register_module()
class ChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are 'smooth_l1', 'l1' or 'l2'. Defaults to 'l2'.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
            Defaults to 'mean'.
        loss_src_weight (float): Weight of loss_source. Defaults to l.0.
        loss_dst_weight (float): Weight of loss_target. Defaults to 1.0.
    """

    def __init__(self,
                 mode: str = 'l2',
                 reduction: str = 'mean',
                 loss_src_weight: float = 1.0,
                 loss_dst_weight: float = 1.0) -> None:
        super(ChamferDistance, self).__init__()

        assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        src_weight: Union[Tensor, float] = 1.0,
        dst_weight: Union[Tensor, float] = 1.0,
        reduction_override: Optional[str] = None,
        return_indices: bool = False,
        **kwargs
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Forward function of loss calculation.

        Args:
            source (Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (Tensor | float):
                Weight of source loss. Defaults to 1.0.
            dst_weight (Tensor | float):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[Tensor]: If ``return_indices=True``, return losses of
                source and target with their corresponding indices in the
                order of ``(loss_source, loss_target, indices1, indices2)``.
                If ``return_indices=False``, return
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = chamfer_distance(
            source, target, src_weight, dst_weight, self.mode, reduction)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        if return_indices:
            return loss_source, loss_target, indices1, indices2
        else:
            return loss_source, loss_target
