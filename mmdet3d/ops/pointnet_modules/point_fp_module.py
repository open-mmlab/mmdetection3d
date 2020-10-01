import torch
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from torch import nn as nn
from typing import List

from mmdet3d.ops import three_interpolate, three_nn


class PointFPModule(nn.Module):
    """Point feature propagation module used in PointNets.

    Propagate the features from one set to another.

    Args:
        mlp_channels (list[int]): List of mlp channels.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 mlp_channels: List[int],
                 norm_cfg: dict = dict(type='BN2d')):
        super().__init__()
        self.fp16_enabled = False
        self.mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 1):
            self.mlps.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg))

    @force_fp32()
    def forward(self, target: torch.Tensor, source: torch.Tensor,
                target_feats: torch.Tensor,
                source_feats: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            target (Tensor): (B, n, 3) tensor of the xyz positions of
                the target features.
            source (Tensor): (B, m, 3) tensor of the xyz positions of
                the source features.
            target_feats (Tensor): (B, C1, n) tensor of the features to be
                propagated to.
            source_feats (Tensor): (B, C2, m) tensor of features
                to be propagated.

        Return:
            Tensor: (B, M, N) M = mlp[-1], tensor of the target features.
        """
        if source is not None:
            dist, idx = three_nn(target, source)
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm

            interpolated_feats = three_interpolate(source_feats, idx, weight)
        else:
            interpolated_feats = source_feats.expand(*source_feats.size()[0:2],
                                                     target.size(1))

        if target_feats is not None:
            new_features = torch.cat([interpolated_feats, target_feats],
                                     dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlps(new_features)

        return new_features.squeeze(-1)
