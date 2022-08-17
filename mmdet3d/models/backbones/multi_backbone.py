# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.models.builder import build_backbone
from mmdet3d.registry import MODELS


@MODELS.register_module()
class MultiBackbone(BaseModule):
    """MultiBackbone with different configs.

    Args:
        num_streams (int): The number of backbones.
        backbones (list or dict): A list of backbone configs.
        aggregation_mlp_channels (list[int]): Specify the mlp layers
            for feature aggregation.
        conv_cfg (dict): Config dict of convolutional layers.
        norm_cfg (dict): Config dict of normalization layers.
        act_cfg (dict): Config dict of activation layers.
        suffixes (list): A list of suffixes to rename the return dict
            for each backbone.
    """

    def __init__(self,
                 num_streams,
                 backbones,
                 aggregation_mlp_channels=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01),
                 act_cfg=dict(type='ReLU'),
                 suffixes=('net0', 'net1'),
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(backbones, dict) or isinstance(backbones, list)
        if isinstance(backbones, dict):
            backbones_list = []
            for ind in range(num_streams):
                backbones_list.append(copy.deepcopy(backbones))
            backbones = backbones_list

        assert len(backbones) == num_streams
        assert len(suffixes) == num_streams

        self.backbone_list = nn.ModuleList()
        # Rename the ret_dict with different suffixs.
        self.suffixes = suffixes

        out_channels = 0

        for backbone_cfg in backbones:
            out_channels += backbone_cfg['fp_channels'][-1][-1]
            self.backbone_list.append(build_backbone(backbone_cfg))

        # Feature aggregation layers
        if aggregation_mlp_channels is None:
            aggregation_mlp_channels = [
                out_channels, out_channels // 2,
                out_channels // len(self.backbone_list)
            ]
        else:
            aggregation_mlp_channels.insert(0, out_channels)

        self.aggregation_layers = nn.Sequential()
        for i in range(len(aggregation_mlp_channels) - 1):
            self.aggregation_layers.add_module(
                f'layer{i}',
                ConvModule(
                    aggregation_mlp_channels[i],
                    aggregation_mlp_channels[i + 1],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs from multiple backbones.

                - fp_xyz[suffix] (list[torch.Tensor]): The coordinates of
                  each fp features.
                - fp_features[suffix] (list[torch.Tensor]): The features
                  from each Feature Propagate Layers.
                - fp_indices[suffix] (list[torch.Tensor]): Indices of the
                  input points.
                - hd_feature (torch.Tensor): The aggregation feature
                  from multiple backbones.
        """
        ret = {}
        fp_features = []
        for ind in range(len(self.backbone_list)):
            cur_ret = self.backbone_list[ind](points)
            cur_suffix = self.suffixes[ind]
            fp_features.append(cur_ret['fp_features'][-1])
            cur_ret_new = dict()
            if cur_suffix != '':
                for k in cur_ret.keys():
                    cur_ret_new[k + '_' + cur_suffix] = cur_ret[k]
            ret.update(cur_ret_new)

        # Combine the features here
        hd_feature = torch.cat(fp_features, dim=1)
        hd_feature = self.aggregation_layers(hd_feature)
        ret['hd_feature'] = hd_feature
        return ret
