import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.utils import get_edge_indices


class EdgeFusionModule(BaseModule):
    """Edge Fusion Module for feature map.

    Args:
        num_classes (int): Number of classes.
        feat_channels (int): Number of channels in feature map
            during edge feature fusion.
        kernel_size (int, optional): Kernel size of convolution.
            Default: 3.
        act_cfg (dict, optional): Config of activation.
            Default: dict(type='ReLU').
        norm_cfg (dict, optional): Config of normalization.
            Default: dict(type='BN')).
    """

    def __init__(self,
                 num_classes,
                 feat_channels,
                 kernel_size=3,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.edge_cls_convs = nn.Sequential(
            ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg='conv1d',
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                padding_mode='replicate'),
            nn.Conv1d(feat_channels, num_classes, kernel_size=1))
        self.edge_reg_convs = nn.Sequential(
            ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg='conv1d',
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                padding_mode='replicate,'),
            nn.Conv1d(feat_channels, 2, kernel_size=1))
        self.feat_channels = feat_channels

    def forward(self, features, fused_features, img_metas):
        """Forward pass.

        Args:
            features (list[torch.Tensor]): Different representative features
                for fusion.
            fused_features (list[torch.Tensor]): Different representative
                features to be fused.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            list[torch.Tensor]: List of fused feature maps.
        """
        bs = features.shape[0]
        edge_indices_list = get_edge_indices(img_metas, device=features.device)
        edge_lens_list = [
            edge_indices.shape[0] for edge_indices in edge_indices_list
        ]
        max_edge_lens = max(edge_lens_list)
        batch_edge_indices = features.new_zeros((bs, max_edge_lens, 2))
        # assert len(features) == len(out_features)
        for i in range(bs):
            batch_edge_indices[i, :edge_lens_list[i]] = edge_indices_list[i]

        # normalize
        grid_edge_indices = batch_edge_indices.view(bs, -1, 1, 2).float()
        grid_edge_indices[..., 0] = \
            grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1
        grid_edge_indices[..., 1] = \
            grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1

        # apply edge fusion for both offset and heatmap
        feature_for_fusion = torch.cat(features, dim=1)
        edge_features = F.grid_sample(
            feature_for_fusion, grid_edge_indices,
            align_corners=True).squeeze(-1)

        edge_cls_feature = edge_features[:, :self.feat_channels, ...]
        edge_reg_feature = edge_features[:, self.feat_channels:, ...]
        edge_cls_output = self.edge_cls_convs(edge_cls_feature)
        edge_reg_output = self.edge_reg_convs(edge_reg_feature)

        for k in range(bs):
            edge_indice_k = edge_indices_list[k]
            fused_features[0][k, :, edge_indice_k[:, 1],
                              edge_indice_k[:, 0]] += edge_cls_output[
                                  k, :, :edge_lens_list[k]]
            fused_features[1][k, :, edge_indice_k[:, 1],
                              edge_indice_k[:, 0]] += edge_reg_output[
                                  k, :, :edge_lens_list[k]]

        return fused_features
