# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.utils import ConfigType


class EdgeFusionModule(BaseModule):
    """Edge Fusion Module for feature map.

    Args:
        out_channels (int): The number of output channels.
        feat_channels (int): The number of channels in feature map
            during edge feature fusion.
        kernel_size (int): Kernel size of convolution. Defaults to 3.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d').
    """

    def __init__(
        self,
        out_channels: int,
        feat_channels: int,
        kernel_size: int = 3,
        act_cfg: ConfigType = dict(type='ReLU'),
        norm_cfg: ConfigType = dict(type='BN1d')
    ) -> None:
        super(EdgeFusionModule, self).__init__()
        self.edge_convs = nn.Sequential(
            ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv1d(feat_channels, out_channels, kernel_size=1))
        self.feat_channels = feat_channels

    def forward(self, features: Tensor, fused_features: Tensor,
                edge_indices: Tensor, edge_lens: List[int], output_h: int,
                output_w: int) -> Tensor:
        """Forward pass.

        Args:
            features (Tensor): Different representative features for fusion.
            fused_features (Tensor): Different representative features
                to be fused.
            edge_indices (Tensor): Batch image edge indices.
            edge_lens (List[int]): List of edge length of each image.
            output_h (int): Height of output feature map.
            output_w (int): Width of output feature map.

        Returns:
            Tensor: Fused feature maps.
        """
        batch_size = features.shape[0]
        # normalize
        grid_edge_indices = edge_indices.view(batch_size, -1, 1, 2).float()
        grid_edge_indices[..., 0] = \
            grid_edge_indices[..., 0] / (output_w - 1) * 2 - 1
        grid_edge_indices[..., 1] = \
            grid_edge_indices[..., 1] / (output_h - 1) * 2 - 1

        # apply edge fusion
        edge_features = F.grid_sample(
            features, grid_edge_indices, align_corners=True).squeeze(-1)
        edge_output = self.edge_convs(edge_features)

        for k in range(batch_size):
            edge_indice_k = edge_indices[k, :edge_lens[k]]
            fused_features[k, :, edge_indice_k[:, 1],
                           edge_indice_k[:, 0]] += edge_output[
                               k, :, :edge_lens[k]]

        return fused_features
