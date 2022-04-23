# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn

from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from .spconv import spconv2_is_avalible

if spconv2_is_avalible:
    from spconv.pytorch import SparseModule, SparseSequential
else:
    from mmcv.ops import SparseModule, SparseSequential


class SparseBottleneck(Bottleneck, SparseModule):
    """Sparse bottleneck block for PartA^2.

    Bottleneck block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int, optional): stride of the first block. Default: 1.
        downsample (Module, optional): down sample module for block.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None.
        norm_cfg (dict, optional): dictionary to construct and config norm
            layer. Default: dict(type='BN').
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None):

        SparseModule.__init__(self)
        Bottleneck.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        if spconv2_is_avalible:
            identity = x.features

            out = self.conv1(x)
            out = out.replace_feature(self.bn1(out.features))
            out = out.replace_feature(self.relu(out.features))

            out = self.conv2(out)
            out = out.replace_feature(self.bn2(out.features))
            out = out.replace_feature(self.relu(out.features))

            out = self.conv3(out)
            out = out.replace_feature(self.bn3(out.features))

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out.replace_feature(out.features + identity)
            out = out.replace_feature(self.relu(out.features))
        else:
            identity = x.features

            out = self.conv1(x)
            out.features = self.bn1(out.features)
            out.features = self.relu(out.features)

            out = self.conv2(out)
            out.features = self.bn2(out.features)
            out.features = self.relu(out.features)

            out = self.conv3(out)
            out.features = self.bn3(out.features)

            if self.downsample is not None:
                identity = self.downsample(x)

            out.features += identity
            out.features = self.relu(out.features)

        return out


class SparseBasicBlock(BasicBlock, SparseModule):
    """Sparse basic block for PartA^2.

    Sparse basic block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int, optional): stride of the first block. Default: 1.
        downsample (Module, optional): down sample module for block.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None.
        norm_cfg (dict, optional): dictionary to construct and config norm
            layer. Default: dict(type='BN').
    """

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None):
        SparseModule.__init__(self)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        if spconv2_is_avalible:
            out = self.conv1(x)
            out = out.replace_feature(self.norm1(out.features))
            out = out.replace_feature(self.relu(out.features))

            out = self.conv2(out)
            out = out.replace_feature(self.norm2(out.features))

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out.replace_feature(out.features + identity)
            out = out.replace_feature(self.relu(out.features))
        else:
            out = self.conv1(x)
            out.features = self.norm1(out.features)
            out.features = self.relu(out.features)

            out = self.conv2(out)
            out.features = self.norm2(out.features)

            if self.downsample is not None:
                identity = self.downsample(x)

            out.features += identity
            out.features = self.relu(out.features)

        return out


def make_sparse_convmodule(in_channels,
                           out_channels,
                           kernel_size,
                           indice_key,
                           stride=1,
                           padding=0,
                           conv_type='SubMConv3d',
                           norm_cfg=None,
                           order=('conv', 'norm', 'act')):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").

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
