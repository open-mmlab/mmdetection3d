# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def monoflex_iou_loss(pred, target, mode='iou', eps=1e-7):
    """FCOS-style IoU loss.

    Computing the FCOS-style IoU loss between a set of predicted
    bboxes and target bboxes. The loss is calculated as negative
    log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of FCOS-style
            format (left, top, right, bottom),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        mode (str, optional): Loss scaling mode, including
            "linear_iou", "iou" and "giou".
            Default: 'iou'
        eps (float, optional): Eps to avoid log(0). Default: 1e-7.

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear_iou', 'iou', 'giou']

    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(
        pred_right, target_right)
    g_w_intersect = torch.max(pred_left, target_left) + torch.max(
        pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
        pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
        pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect + eps
    area_intersect = w_intersect * h_intersect

    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    if mode == 'linear_iou':
        loss = 1 - ious
    elif mode == 'iou':
        loss = -ious.log()
    elif mode == 'giou':
        loss = 1 - gious
    else:
        raise NotImplementedError

    return loss


@LOSSES.register_module()
class MonoFlexIoULoss(nn.Module):
    """FCOS-style IoU Loss for MonoFlex.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float, optional): Eps to avoid log(0). Default: 1e-7.
        reduction (str, optional): Options are "none", "mean" and "sum".
            Default: 'means'.
        loss_weight (float, optional): Weight of loss.
            Default: 1.0.
        mode (str, optional): Loss scaling mode, including
            "linear_iou", "iou" and "giou".
            Default: 'iou'.
    """

    def __init__(self,
                 eps=1e-7,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='iou'):
        super(MonoFlexIoULoss, self).__init__()
        assert mode in ['linear_iou', 'iou', 'giou']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * monoflex_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
