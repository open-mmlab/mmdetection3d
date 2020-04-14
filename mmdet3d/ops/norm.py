import torch.nn as nn

from mmdet.ops.norm import norm_cfg
from .sync_bn import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d

norm_cfg.update({
    'BN1d': ('bn', nn.BatchNorm1d),
    'naiveSyncBN2d': ('bn', NaiveSyncBatchNorm2d),
    'naiveSyncBN1d': ('bn', NaiveSyncBatchNorm1d),
})
