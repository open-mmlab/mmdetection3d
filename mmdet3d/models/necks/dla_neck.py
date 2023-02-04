# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS


def fill_up_weights(up):
    """Simulated bilinear upsampling kernel.

    Args:
        up (nn.Module): ConvTranspose2d module.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUpsample(BaseModule):
    """Iterative Deep Aggregation (IDA) Upsampling module to upsample features
    of different scales to a similar scale.

    Args:
        out_channels (int): Number of output channels for DeformConv.
        in_channels (List[int]): List of input channels of multi-scale
            feature maps.
        kernel_sizes (List[int]): List of size of the convolving
            kernel of different scales.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        use_dcn (bool, optional): If True, use DCNv2. Default: True.
    """

    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_sizes,
        norm_cfg=None,
        use_dcn=True,
        init_cfg=None,
    ):
        super(IDAUpsample, self).__init__(init_cfg)
        self.use_dcn = use_dcn
        self.projs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.nodes = nn.ModuleList()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            up_kernel_size = int(kernel_sizes[i])
            proj = ConvModule(
                in_channel,
                out_channels,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            node = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            up = build_conv_layer(
                dict(type='deconv'),
                out_channels,
                out_channels,
                up_kernel_size * 2,
                stride=up_kernel_size,
                padding=up_kernel_size // 2,
                output_padding=0,
                groups=out_channels,
                bias=False)

            self.projs.append(proj)
            self.ups.append(up)
            self.nodes.append(node)

    def forward(self, mlvl_features, start_level, end_level):
        """Forward function.

        Args:
            mlvl_features (list[torch.Tensor]): Features from multiple layers.
            start_level (int): Start layer for feature upsampling.
            end_level (int): End layer for feature upsampling.
        """
        for i in range(start_level, end_level - 1):
            upsample = self.ups[i - start_level]
            project = self.projs[i - start_level]
            mlvl_features[i + 1] = upsample(project(mlvl_features[i + 1]))
            node = self.nodes[i - start_level]
            mlvl_features[i + 1] = node(mlvl_features[i + 1] +
                                        mlvl_features[i])


class DLAUpsample(BaseModule):
    """Deep Layer Aggregation (DLA) Upsampling module for different scales
    feature extraction, upsampling and fusion, It consists of groups of
    IDAupsample modules.

    Args:
        start_level (int): The start layer.
        channels (List[int]): List of input channels of multi-scale
            feature maps.
        scales(List[int]): List of scale of different layers' feature.
        in_channels (NoneType, optional): List of input channels of
            different scales. Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 start_level,
                 channels,
                 scales,
                 in_channels=None,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLAUpsample, self).__init__(init_cfg)
        self.start_level = start_level
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUpsample(channels[j], in_channels[j:],
                            scales[j:] // scales[j], norm_cfg, use_dcn))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, mlvl_features):
        """Forward function.

        Args:
            mlvl_features(list[torch.Tensor]): Features from multi-scale
                layers.

        Returns:
            tuple[torch.Tensor]: Up-sampled features of different layers.
        """
        outs = [mlvl_features[-1]]
        for i in range(len(mlvl_features) - self.start_level - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(mlvl_features, len(mlvl_features) - i - 2, len(mlvl_features))
            outs.insert(0, mlvl_features[-1])
        return outs


@MODELS.register_module()
class DLANeck(BaseModule):
    """DLA Neck.

    Args:
        in_channels (list[int], optional): List of input channels
            of multi-scale feature map.
        start_level (int, optional): The scale level where upsampling
            starts. Default: 2.
        end_level (int, optional): The scale level where upsampling
            ends. Default: 5.
        norm_cfg (dict, optional): Config dict for normalization
            layer. Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLANeck, self).__init__(init_cfg)
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUpsample(
            start_level=self.start_level,
            channels=in_channels[self.start_level:],
            scales=scales,
            norm_cfg=norm_cfg,
            use_dcn=use_dcn)
        self.ida_up = IDAUpsample(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)], norm_cfg,
            use_dcn)

    def forward(self, x):
        mlvl_features = [x[i] for i in range(len(x))]
        mlvl_features = self.dla_up(mlvl_features)
        outs = []
        for i in range(self.end_level - self.start_level):
            outs.append(mlvl_features[i].clone())
        self.ida_up(outs, 0, len(outs))
        return [outs[-1]]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
