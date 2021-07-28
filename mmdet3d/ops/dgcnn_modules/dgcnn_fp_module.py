import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn
from typing import List

from .builder import FA_MODULES


@FA_MODULES.register_module()
class DGCNNFAModule(BaseModule):
    """Point feature aggregation module used in DGCNN.

    Aggregate all the features of points.

    Args:
        mlp_channels (list[int]): List of mlp channels.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 mlp_channels: List[int],
                 norm_cfg: dict = dict(type='BN1d'),
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
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

    @force_fp32()
    def forward(self, points: List[torch.Tensor]) -> torch.Tensor:
        """forward.

        Args:
            points (List[Tensor]): tensor of the features to be aggregated.

        Return:
            Tensor: (B, N, M) M = mlp[-1], tensor of the output points.
        """

        if points is not None and len(points) > 1:
            new_points = torch.cat(points[1:], dim=-1)
            new_points = new_points.transpose(1, 2).contiguous()  # (B, C, N)
            new_points_copy = new_points

            for i in range(len(self.mlps)):
                new_points = self.mlps[i](new_points)

            new_fa_points = new_points.max(dim=-1, keepdim=True)[0]
            new_fa_points = new_fa_points.repeat(1, 1, new_points.shape[-1])

            new_points = torch.cat([new_points_copy, new_fa_points], dim=1)
            new_points = new_points.transpose(1, 2).contiguous()
        else:
            new_points = points

        return new_points
