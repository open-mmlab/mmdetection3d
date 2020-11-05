import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES


def iou_loss_axis_aligned(center_preds,
                          size_preds,
                          center_targets,
                          size_targets,
                          weight=1.0,
                          reduction='mean'):
    """Calculate the IoU loss (1-IoU) of two axis aligned bounding boxes.

    Args:
        center_preds (torch.Tensor): Center predictions with shape [B, N, 3].
        size_preds (torch.Tensor): Size predictions with shape [B, N, 3].
        center_targets (torch.Tensor): Center targets (gt) \
            with shape [B, N, 3].
        size_targets (torch.Tensor): Size targets with shape [B, N, 3].
        weight (torch.Tensor or float): Weight of loss.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    size_preds = torch.clamp(size_preds, 0)
    max_a = center_preds + size_preds / 2
    max_b = center_targets + size_targets / 2
    min_a = center_preds - size_preds / 2
    min_b = center_targets - size_targets / 2

    max_min = torch.max(min_a, min_b)
    min_max = torch.min(max_a, max_b)
    vol_a = size_preds.prod(dim=2)
    vol_b = size_targets.prod(dim=2)
    diff = torch.clamp(min_max - max_min, min=0)
    intersection = diff.prod(dim=2)
    union = vol_a + vol_b - intersection
    iou_axis_aligned = 1.0 * intersection / (union + 1e-8)

    iou_loss = (1 - iou_axis_aligned) * weight
    if reduction == 'sum':
        iou_loss = torch.sum(iou_loss)
    elif reduction == 'mean':
        iou_loss = torch.mean(iou_loss)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return iou_loss


@LOSSES.register_module()
class IoULossAxisAligned(nn.Module):
    """Calculate the IoU loss (1-IoU) of two axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float): Weight of loss.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
    ):
        super(IoULossAxisAligned, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                center_preds,
                size_preds,
                center_targets,
                size_targets,
                weight=1.0,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            center_preds (torch.Tensor): Center predictions \
                with shape [B, N, 3].
            size_preds (torch.Tensor): Size predictions with shape [B, N, 3].
            center_targets (torch.Tensor): Center targets (gt) \
                with shape [B, N, 3].
            size_targets (torch.Tensor): Size targets with shape [B, N, 3].
            weight (torch.Tensor or float): Weight of loss.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        return iou_loss_axis_aligned(
            center_preds,
            size_preds,
            center_targets,
            size_targets,
            weight=weight,
            reduction=reduction) * self.loss_weight
