# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from mmdet.models import NECKS


@NECKS.register_module()
class LSSFPN(nn.Module):
    r"""Lift-Splat-Shoot FPN.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_

    Args:
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of output feature.
        up_scale (int): Scale factor of up sampling between two input features.
        input_feat_indexes (tuple(int,int)]): Specify the indexes of the
            selected feature in the input feature list.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        up_scale_output (int): The up-sampling scale of the output.
        use_input_conv (bool): Whether to use 1x1 conv to downscale the feature
            channels from in_channels to out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 upsampling_scale=4,
                 input_feat_indexes=(0, 2),
                 norm_cfg=dict(type='BN'),
                 upsampling_scale_output=2,
                 use_input_conv=False):
        super().__init__()
        self.input_feat_indexes = input_feat_indexes
        assert len(input_feat_indexes) == 2
        self.output_upsampling = upsampling_scale_output is not None
        self.up = nn.Upsample(
            scale_factor=upsampling_scale, mode='bilinear', align_corners=True)
        channel_factor = 2 if self.output_upsampling else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channel_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channel_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channel_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channel_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channel_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channel_factor,
                out_channels * channel_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channel_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.output_upsampling:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=upsampling_scale_output,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channel_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )

    def forward(self, feats):
        """Fuse the selected features and up-sample it to the target scale.

        Args:
            feats (list(torch.tensor)): Input feature list.

        Returns:
            torch.tensor: Fuse feature in shape (B, C, H, W)
        """
        x2 = feats[self.input_feat_indexes[0]]
        x1 = feats[self.input_feat_indexes[1]]
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.output_upsampling:
            x = self.up2(x)
        return [x]
