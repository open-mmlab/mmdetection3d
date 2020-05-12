import torch
import torch.nn as nn
from mmcv.cnn import (build_norm_layer, build_upsample_layer, constant_init,
                      is_norm, kaiming_init)

from mmdet.models import NECKS
from .. import builder


@NECKS.register_module()
class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps
        out_channels (list[int]): Output channels of feature maps
        upsample_strides (list[int]): Strides used to upsample the feature maps
        norm_cfg (dict): Config dict of normalization layers
        upsample_cfg (dict): Config dict of upsample layers
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False)):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            upsample_layer = build_upsample_layer(
                upsample_cfg,
                in_channels=in_channels[i],
                out_channels=out_channel,
                kernel_size=upsample_strides[i],
                stride=upsample_strides[i])
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif is_norm(m):
                constant_init(m, 1)

    def forward(self, x):
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


@NECKS.register_module()
class SECONDFusionFPN(SECONDFPN):
    """FPN used in multi-modality SECOND/PointPillars

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps
        out_channels (list[int]): Output channels of feature maps
        upsample_strides (list[int]): Strides used to upsample the feature maps
        norm_cfg (dict): Config dict of normalization layers
        upsample_cfg (dict): Config dict of upsample layers
        downsample_rates (list[int]): The downsample rate of feature map in
            comparison to the original voxelization input
        fusion_layer (dict): Config dict of fusion layers
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 downsample_rates=[40, 8, 8],
                 fusion_layer=None):
        super(SECONDFusionFPN,
              self).__init__(in_channels, out_channels, upsample_strides,
                             norm_cfg, upsample_cfg)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        self.downsample_rates = downsample_rates

    def forward(self,
                x,
                coors=None,
                points=None,
                img_feats=None,
                img_meta=None):
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        if (self.fusion_layer is not None and img_feats is not None):
            downsample_pts_coors = torch.zeros_like(coors)
            downsample_pts_coors[:, 0] = coors[:, 0]
            downsample_pts_coors[:, 1] = (
                coors[:, 1] / self.downsample_rates[0])
            downsample_pts_coors[:, 2] = (
                coors[:, 2] / self.downsample_rates[1])
            downsample_pts_coors[:, 3] = (
                coors[:, 3] / self.downsample_rates[2])
            # fusion for each point
            out = self.fusion_layer(img_feats, points, out,
                                    downsample_pts_coors, img_meta)
        return [out]
