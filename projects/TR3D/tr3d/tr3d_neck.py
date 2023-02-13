# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/tr3d/blob/master/mmdet3d/models/necks/tr3d_neck.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

from torch import nn

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class TR3DNeck(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(TR3DNeck, self).__init__()
        self._init_layers(in_channels[1:], out_channels)

    def _init_layers(self, in_channels, out_channels):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))
                self.__setattr__(
                    f'out_block_{i}',
                    make_block(in_channels[i], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        x = x[1:]
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
                out = self.__getattr__(f'out_block_{i}')(x)
                outs.append(out)
        return outs[::-1]


def make_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels,
                                kernel_size=kernel_size, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_down_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                stride=2, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_up_block(in_channels, out_channels, generative=False):
    conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
        else ME.MinkowskiConvolutionTranspose
    return nn.Sequential(
        conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))
