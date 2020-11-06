import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def iou_loss_axis_aligned(pred, target, eps=1e-6):
    """Calculate the IoU loss (1-IoU) of two set of axis aligned bounding
    boxes. Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [..., 3].
        target (torch.Tensor): Bbox targets (gt) with shape [..., 3].

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """

    (min_a, max_a) = torch.split(pred, 3, dim=-1)
    (min_b, max_b) = torch.split(target, 3, dim=-1)

    max_min = torch.max(min_a, min_b)
    min_max = torch.min(max_a, max_b)
    vol_a = (max_a - min_a).prod(dim=-1)
    vol_b = (max_b - min_b).prod(dim=-1)
    diff = torch.clamp(min_max - max_min, min=0)
    intersection = diff.prod(dim=-1)
    union = vol_a + vol_b - intersection
    eps = union.new_tensor([eps])
    iou_axis_aligned = intersection / torch.max(union, eps)

    iou_loss = 1 - iou_axis_aligned
    return iou_loss


@LOSSES.register_module()
class IoULossAxisAligned(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(IoULossAxisAligned, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
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
            pred (torch.Tensor): Bbox predictions with shape [..., 3].
            target (torch.Tensor): Bbox targets (gt) with shape [..., 3].
            weight (torch.Tensor|float, optional): Weight of loss. \
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        return iou_loss_axis_aligned(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction) * self.loss_weight
