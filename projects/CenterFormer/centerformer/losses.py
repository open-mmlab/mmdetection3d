# modify from https://github.com/TuSimple/centerformer/blob/master/det3d/models/losses/centernet_loss.py # noqa

import torch
from torch import nn

from mmdet3d.registry import MODELS


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


@MODELS.register_module()
class FastFocalLoss(nn.Module):
    """Reimplemented focal loss, exactly the same as the CornerNet version.

    Faster and costs much less memory.
    """

    def __init__(self, focal_factor=2):
        super(FastFocalLoss, self).__init__()
        self.focal_factor = focal_factor

    def forward(self, out, target, ind, mask, cat):
        '''
        Args:
            out, target: B x C x H x W
            ind, mask: B x M
            cat (category id for peaks): B x M
        '''
        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, self.focal_factor) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(
            1 - pos_pred, self.focal_factor) * mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos
