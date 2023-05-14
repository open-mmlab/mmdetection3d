# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.utils import ConfigType, OptConfigType
from .torchsparse import IS_TORCHSPARSE_AVAILABLE

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse.nn as spnn
    from torchsparse.tensor import SparseTensor
else:
    SparseTensor = None


class TorchSparseConvModule(BaseModule):
    """A torchsparse conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        bias (bool): Whether use bias in conv. Defaults to False.
        transposed (bool): Whether use transposed convolution operator.
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): The config of normalization.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False,
                 norm_cfg: ConfigType = dict(type='TorchSparseBN'),
                 act_cfg: ConfigType = dict(
                     type='TorchSparseReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg)
        layers = [
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride,
                        dilation, bias, transposed)
        ]
        if norm_cfg is not None:
            _, norm = build_norm_layer(norm_cfg, out_channels)
            layers.append(norm)
        if act_cfg is not None:
            activation = build_activation_layer(act_cfg)
            layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.net(x)
        return out


class TorchSparseBasicBlock(BaseModule):
    """Torchsparse residual basic block for MinkUNet.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        bias (bool): Whether use bias in conv. Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): The config of normalization.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 stride: Union[int, Sequence[int]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 norm_cfg: ConfigType = dict(type='TorchSparseBN'),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg)
        _, norm1 = build_norm_layer(norm_cfg, out_channels)
        _, norm2 = build_norm_layer(norm_cfg, out_channels)

        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride,
                        dilation, bias), norm1, spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
                bias=bias), norm2)

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Identity()
        else:
            _, norm3 = build_norm_layer(norm_cfg, out_channels)
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=bias), norm3)

        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class TorchSparseBottleneck(BaseModule):
    """Torchsparse residual basic block for MinkUNet.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the second block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        bias (bool): Whether use bias in conv. Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): The config of normalization.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 stride: Union[int, Sequence[int]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 norm_cfg: ConfigType = dict(type='TorchSparseBN'),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg)
        _, norm1 = build_norm_layer(norm_cfg, out_channels)
        _, norm2 = build_norm_layer(norm_cfg, out_channels)
        _, norm3 = build_norm_layer(norm_cfg, out_channels)

        self.net = nn.Sequential(
            spnn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=dilation,
                bias=bias), norm1, spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=dilation,
                bias=bias), norm2, spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=dilation,
                bias=bias), norm3)

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Identity()
        else:
            _, norm4 = build_norm_layer(norm_cfg, out_channels)
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=bias), norm4)

        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.relu(self.net(x) + self.downsample(x))
        return out
