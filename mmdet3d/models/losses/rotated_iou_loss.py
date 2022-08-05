# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import diff_iou_rotated_3d
from torch import nn as nn

from mmdet.models.losses.utils import weighted_loss
from ..builder import LOSSES


@weighted_loss
def rotated_iou_3d_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of rotated bounding boxes.
    Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [N, 7]
            (x, y, z, w, l, h, alpha).
        target (torch.Tensor): Bbox targets (gt) with shape [N, 7]
            (x, y, z, w, l, h, alpha).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    iou_loss = 1 - diff_iou_rotated_3d(pred.unsqueeze(0),
                                       target.unsqueeze(0))[0]
    return iou_loss


@LOSSES.register_module()
class RotatedIoU3DLoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of rotated bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_iou_3d_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss
