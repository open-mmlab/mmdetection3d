# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import nn

from mmdet3d.utils import ConfigType, OptConfigType

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
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='MinkowskiBN'),
                 act_cfg: ConfigType = dict(
                     type='MinkowskiReLU', inplace=True),
                 init_cfg: OptConfigType = None,
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
