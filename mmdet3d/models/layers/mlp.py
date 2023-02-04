# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.utils import ConfigType, OptMultiConfig


class MLP(BaseModule):
    """A simple MLP module.

    Pass features (B, C, N) through an MLP.

    Args:
        in_channels (int): Number of channels of input features.
            Defaults to 18.
        conv_channels (Tuple[int]): Out channels of the convolution.
            Defaults to (256, 256).
        conv_cfg (:obj:`ConfigDict` or dict): Config dict for convolution
            layer. Defaults to dict(type='Conv1d').
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channel: int = 18,
                 conv_channels: Tuple[int] = (256, 256),
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(MLP, self).__init__(init_cfg=init_cfg)
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

    def forward(self, img_features: Tensor) -> Tensor:
        return self.mlp(img_features)
