# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn


class MLP(BaseModule):
    """A simple MLP module.

    Pass features (B, C, N) through an MLP.

    Args:
        in_channels (int, optional): Number of channels of input features.
            Default: 18.
        conv_channels (tuple[int], optional): Out channels of the convolution.
            Default: (256, 256).
        conv_cfg (dict, optional): Config of convolution.
            Default: dict(type='Conv1d').
        norm_cfg (dict, optional): Config of normalization.
            Default: dict(type='BN1d').
        act_cfg (dict, optional): Config of activation.
            Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channel=18,
                 conv_channels=(256, 256),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.mlp = nn.Sequential()
        prev_channels = in_channel
        for i, conv_channel in enumerate(conv_channels):
            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    prev_channels,
                    conv_channels[i],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[i]

    def forward(self, img_features):
        return self.mlp(img_features)
