from __future__ import absolute_import, division, print_function

import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger


def dla_build_norm_layer(cfg, num_features):
    """Build normalization layer specially designed for DLANet.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.


    Returns:
        Function: Build normalization layer in mmcv.
    """
    cfg_ = cfg.copy()
    if cfg_['type'] == 'GN':
        if num_features % 32 == 0:
            return build_norm_layer(cfg_, num_features)
        else:
            cfg_['num_groups'] = cfg_['num_groups'] // 2
            return build_norm_layer(cfg_, num_features)
    else:
        return build_norm_layer(cfg_, num_features)


class BasicBlock(nn.Module):
    """BasicBlock in DLANet.

    Args:
        inplanes (int): Input feature channel.
        planes (int): Output feature channel.
        norm_cfg (dict): Dictionary to construct and config
            norm layer.
        conv_cfg (dict): Dictionary to construct and config
            conv layer.
        stride (int, optional): Conv stride. Default: 1.
        dilation (int, optional): Conv dilation. Default: 1.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 norm_cfg,
                 conv_cfg,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.norm1 = dla_build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.norm2 = dla_build_norm_layer(norm_cfg, planes)[1]
        self.stride = stride

    def forward(self, x, residual=None):
        """Forward function."""

        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, conv_cfg,
                 kernel_size, residual):
        super(Root, self).__init__()
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.norm = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, feat_list):
        """Forward function.

        Args:
            feat_list (list[torch.Tensor]): Output features from
                multiple layers.
        """
        children = feat_list
        x = self.conv(torch.cat(feat_list, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 conv_cfg,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                stride,
                dilation=dilation)
            self.tree2 = block(
                out_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                1,
                dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, norm_cfg, conv_cfg,
                             root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    1,
                    stride=1,
                    bias=False),
                dla_build_norm_layer(norm_cfg, out_channels)[1])

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            feat_list = [x2, x1] + children
            x = self.root(feat_list)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLANet(nn.Module):
    r"""`DLA backbone <https://arxiv.org/abs/1707.06484>`_.

    Args:
        depth (int, optional): Depth of DLA. Default: 34.
        in_channels (int, optional): Number of input image channels.
            Default: 3.
        norm_cfg (dict, optional): Dictionary to construct and config
            norm layer. Default: None.
        conv_cfg (dict, optional): Dictionary to construct and config
            conv layer. Default: None.
        layer_level_root (list[bool], optional): Whether to apply
            level_root in each DLA layer, this is only used for
            tree levels. Default: (False, True, True, True).
        residual_root (bool, optional): Whether to use residual
            in root layer. Default: False.
    """
    arch_settings = {
        34: (BasicBlock, (1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512)),
    }

    def __init__(self,
                 depth=34,
                 in_channels=3,
                 norm_cfg=None,
                 conv_cfg=None,
                 layer_level_root=(False, True, True, True),
                 residual_root=False):
        super(DLANet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalida depth {depth} for DLA')
        block, levels, channels = self.arch_settings[depth]
        self.channels = channels
        self.num_levels = len(levels)
        self.base_layer = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                channels[0],
                7,
                stride=1,
                padding=3,
                bias=False),
            dla_build_norm_layer(norm_cfg, channels[0])[1],
            nn.ReLU(inplace=True))

        # DLANet first uses two conv layers then uses several
        # Tree layers
        for i in range(2):
            level_layer = self._make_conv_level(
                channels[0],
                channels[i],
                levels[i],
                norm_cfg,
                conv_cfg,
                stride=i + 1)
            layer_name = f'level{i}'
            self.add_module(layer_name, level_layer)

        for i in range(2, self.num_levels):
            dla_layer = Tree(
                levels[i],
                block,
                channels[i - 1],
                channels[i],
                norm_cfg,
                conv_cfg,
                2,
                level_root=layer_level_root[i - 2],
                root_residual=residual_root)
            layer_name = f'level{i}'
            self.add_module(layer_name, dla_layer)

    def _make_conv_level(self,
                         inplanes,
                         planes,
                         convs,
                         norm_cfg,
                         conv_cfg,
                         stride=1,
                         dilation=1):
        """Conv modules.

        Args:
            inplanes (int): Input feature channel.
            planes (int): Output feature channel.
            convs (int): Number of Conv module.
            norm_cfg (dict): Dictionary to construct and config
                norm layer.
            conv_cfg (dict): Dictionary to construct and config
                conv layer.
            stride (int, optional): Conv stride. Default: 1.
            dilation (int, optional): Conv dilation. Default: 1.
        """
        modules = []
        for i in range(convs):
            modules.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation),
                dla_build_norm_layer(norm_cfg, planes)[1],
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        outs = []
        x = self.base_layer(x)
        for i in range(self.num_levels):
            x = getattr(self, 'level{}'.format(i))(x)
            outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
