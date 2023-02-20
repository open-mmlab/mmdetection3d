# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/resnet.py # noqa
# and mmcv.cnn.ResNet
from typing import List, Union

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    # blocks are used in the static part of MinkResNet
    ME = BasicBlock = Bottleneck = SparseTensor = None

import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class MinkResNet(BaseModule):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int): Resnet stages. Defaults to 4.
        pool (bool): Whether to add max pooling after first conv.
            Defaults to True.
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 in_channels: int,
                 num_stages: int = 4,
                 pool: bool = True):
        super(MinkResNet, self).__init__()
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.`')
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.num_stages = num_stages
        self.pool = pool

        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3)
        # May be BatchNorm is better, but we follow original implementation.
        self.norm1 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(
                kernel_size=2, stride=2, dimension=3)

        for i in range(len(stage_blocks)):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, 64 * 2**i, stage_blocks[i], stride=2))

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int,
                    blocks: int, stride: int) -> nn.Module:
        """Make single level of residual blocks.

        Args:
            block (BasicBlock | Bottleneck): Residual block class.
            planes (int): Number of convolution filters.
            blocks (int): Number of blocks in the layers.
            stride (int): Stride of the first convolutional layer.

        Returns:
            nn.Module: With residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs
