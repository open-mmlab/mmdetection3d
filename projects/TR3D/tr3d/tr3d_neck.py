# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/tr3d/blob/master/mmdet3d/models/necks/tr3d_neck.py # noqa
from typing import List, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TR3DNeck(BaseModule):
    r"""Neck of `TR3D <https://arxiv.org/abs/2302.02858>`_.

    Args:
        in_channels (tuple[int]): Number of channels in input tensors.
        out_channels (int): Number of channels in output tensors.
    """

    def __init__(self, in_channels: Tuple[int], out_channels: int):
        super(TR3DNeck, self).__init__()
        self._init_layers(in_channels[1:], out_channels)

    def _init_layers(self, in_channels: Tuple[int], out_channels: int):
        """Initialize layers.

        Args:
            in_channels (tuple[int]): Number of channels in input tensors.
            out_channels (int): Number of channels in output tensors.
        """
        for i in range(len(in_channels)):
            if i > 0:
                self.add_module(
                    f'up_block_{i}',
                    self._make_block(in_channels[i], in_channels[i - 1], True,
                                     2))
            if i < len(in_channels) - 1:
                self.add_module(
                    f'lateral_block_{i}',
                    self._make_block(in_channels[i], in_channels[i]))
                self.add_module(f'out_block_{i}',
                                self._make_block(in_channels[i], out_channels))

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: List[SparseTensor]) -> List[SparseTensor]:
        """Forward pass.

        Args:
            x (list[SparseTensor]): Features from the backbone.

        Returns:
            List[Tensor]: Output features from the neck.
        """
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

    @staticmethod
    def _make_block(in_channels: int,
                    out_channels: int,
                    generative: bool = False,
                    stride: int = 1) -> nn.Module:
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            generative (bool): Use generative convolution if True.
                Defaults to False.
            stride (int): Stride of the convolution. Defaults to 1.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolution
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                dimension=3), ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))
