# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.losses.utils import weighted_loss
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS


@weighted_loss
def uncertain_smooth_l1_loss(pred: Tensor,
                             target: Tensor,
                             sigma: Tensor,
                             alpha: float = 1.0,
                             beta: float = 1.0) -> Tensor:
    """Smooth L1 loss with uncertainty.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        sigma (Tensor): The sigma for uncertainty.
        alpha (float): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    assert target.numel() > 0
    assert pred.size() == target.size() == sigma.size(), 'The size of pred ' \
        f'{pred.size()}, target {target.size()}, and sigma {sigma.size()} ' \
        'are inconsistent.'
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    loss = torch.exp(-sigma) * loss + alpha * sigma

    return loss


@weighted_loss
def uncertain_l1_loss(pred: Tensor,
                      target: Tensor,
                      sigma: Tensor,
                      alpha: float = 1.0) -> Tensor:
    """L1 loss with uncertainty.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        sigma (Tensor): The sigma for uncertainty.
        alpha (float): The coefficient of log(sigma).
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert target.numel() > 0
    assert pred.size() == target.size() == sigma.size(), 'The size of pred ' \
        f'{pred.size()}, target {target.size()}, and sigma {sigma.size()} ' \
        'are inconsistent.'
    loss = torch.abs(pred - target)
    loss = torch.exp(-sigma) * loss + alpha * sigma
    return loss


@MODELS.register_module()
class UncertainSmoothL1Loss(nn.Module):
    r"""Smooth L1 loss with uncertainty.

    Please refer to `PGD <https://arxiv.org/abs/2107.14160>`_ and
    `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics <https://arxiv.org/abs/1705.07115>`_ for more details.

    Args:
        alpha (float): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float): The weight of loss. Defaults to 1.0
    """

    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super(UncertainSmoothL1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                sigma: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            sigma (Tensor): The sigma for uncertainty.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * uncertain_smooth_l1_loss(
            pred,
            target,
            weight,
            sigma=sigma,
            alpha=self.alpha,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@MODELS.register_module()
class UncertainL1Loss(nn.Module):
    """L1 loss with uncertainty.

    Args:
        alpha (float): The coefficient of log(sigma).
            Defaults to 1.0.
        reduction (str): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float): The weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 alpha: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super(UncertainL1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                sigma: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            sigma (Tensor): The sigma for uncertainty.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * uncertain_l1_loss(
            pred,
            target,
            weight,
            sigma=sigma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox
