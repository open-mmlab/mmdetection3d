# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F


class EdgeFusionModule(BaseModule):
    """Edge Fusion Module for feature map.

    Args:
        out_channels (int): The number of output channels.
        feat_channels (int): The number of channels in feature map
            during edge feature fusion.
        kernel_size (int, optional): Kernel size of convolution.
            Default: 3.
        act_cfg (dict, optional): Config of activation.
            Default: dict(type='ReLU').
        norm_cfg (dict, optional): Config of normalization.
            Default: dict(type='BN1d')).
    """

    def __init__(self,
                 out_channels,
                 feat_channels,
                 kernel_size=3,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN1d')):
        super().__init__()
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

    def forward(self, features, fused_features, edge_indices, edge_lens,
                output_h, output_w):
        """Forward pass.

        Args:
            features (torch.Tensor): Different representative features
                for fusion.
            fused_features (torch.Tensor): Different representative
                features to be fused.
            edge_indices (torch.Tensor): Batch image edge indices.
            edge_lens (list[int]): List of edge length of each image.
            output_h (int): Height of output feature map.
            output_w (int): Width of output feature map.

        Returns:
            torch.Tensor: Fused feature maps.
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
