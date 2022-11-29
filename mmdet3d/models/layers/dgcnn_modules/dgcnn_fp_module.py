# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.utils import ConfigType, OptMultiConfig


class DGCNNFPModule(BaseModule):
    """Point feature propagation module used in DGCNN.

    Propagate the features from one set to another.

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
        super(DGCNNFPModule, self).__init__(init_cfg=init_cfg)
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

    def forward(self, points: Tensor) -> Tensor:
        """Forward.

        Args:
            points (Tensor): (B, N, C) Tensor of the input points.

        Returns:
            Tensor: (B, N, M) M = mlp[-1]. Tensor of the new points.
        """

        if points is not None:
            new_points = points.transpose(1, 2).contiguous()  # (B, C, N)
            new_points = self.mlps(new_points)
            new_points = new_points.transpose(1, 2).contiguous()
        else:
            new_points = points

        return new_points
