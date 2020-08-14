import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CenterPointFocalLoss(nn.Module):
    """Modified focal loss.

    Exactly the same as CornerNet. Runs faster and costs a little bit more
    memory.
    """

    def __init__(self):
        super(CenterPointFocalLoss, self).__init__()

    def forward(self, pred, gt):
        """Forward function of CenterPointFocalLoss.

        Args:
          pred (torch.Tensor): Prediction results with the shape
            of [b x c x h x w].
          gt_regr (torch.Tensor): Ground truth of the predicition
            results with the shape of (batch x c x h x w).

        Returns:
            torch.Tensor: Computed loss.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                                   2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss
