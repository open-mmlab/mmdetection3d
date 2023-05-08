# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmengine import MODELS
from torch import nn

from mmdet3d.utils import OptConfigType
from .spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseModule, SparseSequential


def replace_feature(out: SparseConvTensor,
                    new_features: SparseConvTensor) -> SparseConvTensor:
    if 'replace_feature' in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


# @MODELS.register_module()
# class SparseConvModule(ConvModule, SparseModule):
#     """A sparse conv block that bundles conv/norm/activation layers. It
#     different from ConvModule in that is implemented with submanifold sparse
#     convolution.

#     Args:
#         in_channels (int): Number of channels in the input feature map.
#             Same as that in ``nn._ConvNd``.
#         out_channels (int): Number of channels produced by the convolution.
#             Same as that in ``nn._ConvNd``.
#         kernel_size (int | tuple[int]): Size of the convolving kernel.
#             Same as that in ``nn._ConvNd``.
#         stride (int | tuple[int]): Stride of the convolution.
#             Same as that in ``nn._ConvNd``.
#         padding (int | tuple[int]): Zero-padding added to both sides of
#             the input. Same as that in ``nn._ConvNd``.
#         dilation (int | tuple[int]): Spacing between kernel elements.
#             Same as that in ``nn._ConvNd``.
#         groups (int): Number of blocked connections from input channels to
#             output channels. Same as that in ``nn._ConvNd``.
#         bias (bool | str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if `norm_cfg` is None,
#               otherwise
#             False. Default: "auto".
#         indice_key (str): Indice key for spconv. Default: None.
#         conv_cfg (dict): Config dict for convolution layer. Default: None,
#             which means using conv2d.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (dict): Config dict for activation layer.
#             Default: dict(type='ReLU').
#         inplace (bool): Whether to use inplace mode for activation.
#             Default: True.
#         order (tuple[str]): The order of conv/norm/activation layers. It is a
#             sequence of "conv", "norm" and "act". Common examples are
#             ("conv", "norm", "act") and ("act", "conv", "norm").
#             Default: ('conv', 'norm', 'act').
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[int, Tuple[int, int]],
#         stride: Union[int, Tuple[int, int]] = 1,
#         padding: Union[int, Tuple[int, int]] = 0,
#         dilation: Union[int, Tuple[int, int]] = 1,
#         groups: int = 1,
#         bias: Union[bool, str] = 'auto',
#         indice_key=None,
#         conv_cfg: ConfigType = dict(type='SubMConv3d'),
#         norm_cfg: ConfigType = dict(type='BN1d'),
#         act_cfg: ConfigType = dict(type='ReLU'),
#         inplace: bool = True,
#         order: tuple = ('conv', 'norm', 'act')) -> None:

#         SparseModule.__init__(self)
#         if conv_cfg is not None:
#             conv_cfg.setdefault('indice_key', indice_key)
#         ConvModule.__init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#             inplace=inplace,
#             order=order)

#     def forward(self,
#                 x: SparseConvTensor,
#                 activate: bool = True,
#                 norm: bool = True) -> SparseConvTensor:
#         for layer in self.order:
#             if layer == 'conv':
#                 if self.with_explicit_padding:
#                     x = self.padding_layer(x)
#                 x = self.conv(x)
#             elif layer == 'norm' and norm and self.with_norm:
#                 x = replace_feature(x, self.norm(x))
#             elif layer == 'act' and activate and self.with_activation:
#                 x = replace_feature(x, self.activate(x))
#         return x


@MODELS.register_module()
class SparseBottleneck(Bottleneck, SparseModule):
    """Sparse bottleneck block for PartA^2.

    Bottleneck block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): Inplanes of block.
        planes (int): Planes of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        downsample (Module, optional): Down sample module for block.
            Defaults to None.
        indice_key (str): Indice key for spconv. Default to None.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
    """

    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: Union[int, Tuple[int]] = 1,
                 downsample: nn.Module = None,
                 indice_key=None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:

        SparseModule.__init__(self)
        if conv_cfg is not None:
            conv_cfg.setdefault('indice_key', indice_key)
        Bottleneck.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv3(out)
        out = replace_feature(out, self.bn3(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


@MODELS.register_module()
class SparseBasicBlock(BasicBlock, SparseModule):
    """Sparse basic block for PartA^2.

    Sparse basic block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): Inplanes of block.
        planes (int): Planes of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        downsample (Module, optional): Down sample module for block.
            Defaults to None.
        indice_key (str): Indice key for spconv. Default to None.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
    """

    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: Union[int, Tuple[int]] = 1,
                 downsample: nn.Module = None,
                 indice_key=None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:
        SparseModule.__init__(self)
        if conv_cfg is not None:
            conv_cfg.setdefault('indice_key', indice_key)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x.features

        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


def make_sparse_convmodule(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int]],
    indice_key: str,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    conv_type: str = 'SubMConv3d',
    norm_cfg: OptConfigType = None,
    order: Tuple[str] = ('conv', 'norm', 'act'),
    **kwargs,
) -> SparseSequential:
    """Make sparse convolution module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of out channels.
        kernel_size (int | Tuple[int]): Kernel size of convolution.
        indice_key (str): The indice key used for sparse tensor.
        stride (int or tuple[int]): The stride of convolution.
        padding (int or tuple[int]): The padding number of input.
        conv_type (str): Sparse conv type in spconv. Defaults to 'SubMConv3d'.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        order (Tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Defaults to ('conv', 'norm', 'act').

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)

    layers = list()
    for layer in order:
        if layer == 'conv':
            if conv_type not in [
                    'SparseInverseConv3d', 'SparseInverseConv2d',
                    'SparseInverseConv1d'
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False))
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        bias=False))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            layers.append(nn.ReLU(inplace=True))

    layers = SparseSequential(*layers)
    return layers
