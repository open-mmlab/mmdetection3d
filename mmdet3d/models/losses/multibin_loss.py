# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmdet.models.builder import LOSSES


def MultiBin_loss(vector_ori, gt_ori, num_dir_bins=4, reduction='mean'):
    # bin1 cls, bin1 offset, bin2 cls, bin2 offst
    gt_ori = gt_ori.view(-1, gt_ori.shape[-1])

    cls_losses = 0
    reg_losses = 0
    reg_cnt = 0
    for i in range(num_dir_bins):
        # bin cls loss
        cls_ce_loss = F.cross_entropy(
            vector_ori[:, (i * 2):(i * 2 + 2)],
            gt_ori[:, i].long(),
            reduction=reduction)
        # regression loss
        valid_mask_i = (gt_ori[:, i] == 1)
        cls_losses += cls_ce_loss.mean()
        if valid_mask_i.sum() > 0:
            s = num_dir_bins * 2 + i * 2
            e = s + 2
            pred_offset = F.normalize(vector_ori[valid_mask_i, s:e])
            reg_loss = \
                F.l1_loss(pred_offset[:, 0],
                          torch.sin(gt_ori[valid_mask_i, num_dir_bins + i]),
                          reduction=reduction) + \
                F.l1_loss(pred_offset[:, 1],
                          torch.cos(gt_ori[valid_mask_i, num_dir_bins + i]),
                          reduction=reduction)

            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

        return cls_losses / num_dir_bins + reg_losses / reg_cnt


@LOSSES.register_module()
class MultiBinLoss(nn.Module):
    """Multi-Bin Loss for orientation.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MultiBinLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, num_dir_bins, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            num_dir_bins (int): Number of bins to encode direction angle.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * MultiBin_loss(
            pred, target, num_dir_bins=num_dir_bins, reduction=reduction)
        return loss_bbox
