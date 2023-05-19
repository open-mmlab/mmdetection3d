# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import nn

from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

try:
    from MinkowskiEngine import (MinkowskiBatchNorm, MinkowskiConvolution,
                                 MinkowskiConvolutionTranspose, MinkowskiReLU,
                                 MinkowskiSyncBatchNorm, SparseTensor)
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    SparseTensor = None
    from mmcv.cnn.resnet import BasicBlock, Bottleneck
    IS_MINKOWSKI_ENGINE_AVAILABLE = False
else:
    MODELS._register_module(MinkowskiConvolution, 'MinkowskiConvNd')
    MODELS._register_module(MinkowskiConvolutionTranspose,
                            'MinkowskiConvNdTranspose')
    MODELS._register_module(MinkowskiBatchNorm, 'MinkowskiBN')
    MODELS._register_module(MinkowskiSyncBatchNorm, 'MinkowskiSyncBN')
    MODELS._register_module(MinkowskiReLU, 'MinkowskiReLU')
    IS_MINKOWSKI_ENGINE_AVAILABLE = True


class MinkowskiConvModule(BaseModule):
    """A minkowski engine conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        bias (bool): Whether to use bias in conv. Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config of conv layer.
            Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): The config of normalization.
            Defaults to dict(type='MinkowskiBN').
        act_cfg (:obj:`ConfigDict` or dict): The config of activation.
            Defaults to dict(type='MinkowskiReLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='MinkowskiBN'),
                 act_cfg: ConfigType = dict(
                     type='MinkowskiReLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(init_cfg)
        layers = []
        if conv_cfg is None:
            conv_cfg = dict(type='MinkowskiConvNd')
        conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            dimension=3)
        layers.append(conv)

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


class MinkowskiBasicBlock(BasicBlock, BaseModule):
    """A wrapper of minkowski engine basic block. It inherits from mmengine's
    `BaseModule` and allows additional keyword arguments.

    Args:
        inplanes (int): In channels of block.
        planes (int): Out channels of block.
        stride (int or Tuple[int]): Stride of the first conv. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        downsample (nn.Module, optional): Residual branch conv module if
            necessary. Defaults to None.
        bn_momentum (float): Momentum of batch norm layer. Defaults to 0.1.
        dimension (int): Dimension of minkowski convolution. Defaults to 3.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 bn_momentum: float = 0.1,
                 dimension: int = 3,
                 init_cfg: OptConfigType = None,
                 **kwargs):
        BaseModule.__init__(self, init_cfg)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            bn_momentum=bn_momentum,
            dimension=dimension)


class MinkowskiBottleneck(Bottleneck, BaseModule):
    """A wrapper of minkowski engine bottleneck block. It inherits from
    mmengine's `BaseModule` and allows additional keyword arguments.

    Args:
        inplanes (int): In channels of block.
        planes (int): Out channels of block.
        stride (int or Tuple[int]): Stride of the second conv. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        downsample (nn.Module, optional): Residual branch conv module if
            necessary. Defaults to None.
        bn_momentum (float): Momentum of batch norm layer. Defaults to 0.1.
        dimension (int): Dimension of minkowski convolution. Defaults to 3.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 bn_momentum: float = 0.1,
                 dimension: int = 3,
                 init_cfg: OptConfigType = None,
                 **kwargs):
        BaseModule.__init__(self, init_cfg)
        Bottleneck.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            bn_momentum=bn_momentum,
            dimension=dimension)
