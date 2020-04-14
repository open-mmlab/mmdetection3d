import logging
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn import Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer


class Empty(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


@NECKS.register_module
class SECONDFPN(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """

    def __init__(self,
                 use_norm=True,
                 in_channels=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__()
        assert len(num_upsample_filters) == len(upsample_strides)
        self.in_channels = in_channels

        if norm_cfg is not None:
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        deblocks = []

        for i, num_upsample_filter in enumerate(num_upsample_filters):
            norm_layer = (
                build_norm_layer(norm_cfg, num_upsample_filter)[1]
                if norm_cfg is not None else Empty)
            deblock = Sequential(
                ConvTranspose2d(
                    in_channels[i],
                    num_upsample_filter,
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                norm_layer,
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # keeping the initiation yields better results
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
        return

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        ups = [deblock(inputs[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        return [x]


@NECKS.register_module
class SECONDFusionFPN(SECONDFPN):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """

    def __init__(self,
                 use_norm=True,
                 in_channels=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 down_sample_rate=[40, 8, 8],
                 fusion_layer=None,
                 cat_points=False):
        super(SECONDFusionFPN, self).__init__(
            use_norm,
            in_channels,
            upsample_strides,
            num_upsample_filters,
            norm_cfg,
        )
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        self.cat_points = cat_points
        self.down_sample_rate = down_sample_rate

    def forward(self,
                inputs,
                coors=None,
                points=None,
                img_feats=None,
                img_meta=None):
        assert len(inputs) == len(self.in_channels)
        ups = [deblock(inputs[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if (self.fusion_layer is not None and img_feats is not None):
            downsample_pts_coors = torch.zeros_like(coors)
            downsample_pts_coors[:, 0] = coors[:, 0]
            downsample_pts_coors[:, 1] = (
                coors[:, 1] / self.down_sample_rate[0])
            downsample_pts_coors[:, 2] = (
                coors[:, 2] / self.down_sample_rate[1])
            downsample_pts_coors[:, 3] = (
                coors[:, 3] / self.down_sample_rate[2])
            # fusion for each point
            x = self.fusion_layer(img_feats, points, x, downsample_pts_coors,
                                  img_meta)
        return [x]
