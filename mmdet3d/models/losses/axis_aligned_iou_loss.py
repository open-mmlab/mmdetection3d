# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.losses.utils import weighted_loss
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.structures import AxisAlignedBboxOverlaps3D


@weighted_loss
def axis_aligned_iou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Calculate the IoU loss (1-IoU) of two set of axis aligned bounding
    boxes. Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (Tensor): Bbox predictions with shape [..., 3].
        target (Tensor): Bbox targets (gt) with shape [..., 3].

    Returns:
        Tensor: IoU loss between predictions and targets.
    """

    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        pred, target, is_aligned=True)
    iou_loss = 1 - axis_aligned_iou
    return iou_loss


@MODELS.register_module()
class AxisAlignedIoULoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super(AxisAlignedIoULoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function of loss calculation.

        Args:
            pred (Tensor): Bbox predictions with shape [..., 3].
            target (Tensor): Bbox targets (gt) with shape [..., 3].
            weight (Tensor, optional): Weight of loss.
                Defaults to None.
            avg_factor (float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            Tensor: IoU loss between predictions and targets.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        return axis_aligned_iou_loss(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction) * self.loss_weight
