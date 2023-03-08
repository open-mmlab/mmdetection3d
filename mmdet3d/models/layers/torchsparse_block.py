# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

from mmengine.model import BaseModule
from torch import nn

from mmdet3d.utils import OptConfigType
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
        transposed (bool): Whether use transposed convolution operator.
            Defaults to False.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        dilation: int = 1,
        bias: bool = False,
        transposed: bool = False,
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg)
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride,
                        dilation, bias, transposed),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.net(x)
        return out


class TorchSparseResidualBlock(BaseModule):
    """Torchsparse residual basic block for MinkUNet.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        dilation: int = 1,
        bias: bool = False,
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg)
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride,
                        dilation, bias),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
                bias=bias),
            spnn.BatchNorm(out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=bias),
                spnn.BatchNorm(out_channels),
            )

        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.relu(self.net(x) + self.downsample(x))
        return out
