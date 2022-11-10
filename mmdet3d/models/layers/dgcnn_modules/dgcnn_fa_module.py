# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.utils import ConfigType, OptMultiConfig


class DGCNNFAModule(BaseModule):
    """Point feature aggregation module used in DGCNN.

    Aggregate all the features of points.

    Args:
        mlp_channels (List[int]): List of mlp channels.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(DGCNNFAModule, self).__init__(init_cfg=init_cfg)
        self.mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 1):
            self.mlps.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    kernel_size=(1, ),
                    stride=(1, ),
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, points: List[Tensor]) -> Tensor:
        """forward.

        Args:
            points (List[Tensor]): Tensor of the features to be aggregated.

        Returns:
            Tensor: (B, N, M) M = mlp[-1]. Tensor of the output points.
        """

        if len(points) > 1:
            new_points = torch.cat(points[1:], dim=-1)
            new_points = new_points.transpose(1, 2).contiguous()  # (B, C, N)
            new_points_copy = new_points

            new_points = self.mlps(new_points)

            new_fa_points = new_points.max(dim=-1, keepdim=True)[0]
            new_fa_points = new_fa_points.repeat(1, 1, new_points.shape[-1])

            new_points = torch.cat([new_fa_points, new_points_copy], dim=1)
            new_points = new_points.transpose(1, 2).contiguous()
        else:
            new_points = points

        return new_points
