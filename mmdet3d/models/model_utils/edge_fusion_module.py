import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F


class EdgeFusionModule(BaseModule):
    """Forward pass.

    Args:
        feats (list[torch.Tensor]): Different representative features.

    Returns:
        tuple[list[torch.Tensor]]: Multi-level class score, bbox \
            and direction predictions.
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
                padding_mode='replicate,'),
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

    def forward(self, features, out_features, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Different representative features.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        assert len(features) == len(out_features)
        bs = features.shape[0]
        edge_indices = torch.stack(
            [img_meta['edge_indices'] for img_meta in img_metas])
        edge_lens = torch.stack(
            [img_meta['edge_len'] for img_meta in img_metas])
        # normalize
        grid_edge_indices = edge_indices.view(bs, -1, 1, 2).float()
        grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (
            self.output_width - 1) * 2 - 1  # -》（-1，1）
        grid_edge_indices[
            ...,
            1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1

        # apply edge fusion for both offset and heatmap
        feature_for_fusion = torch.cat(
            features, dim=1)  # cls_feat + reg_feat，两个concat在一起，一起sample
        edge_features = F.grid_sample(
            feature_for_fusion, grid_edge_indices,
            align_corners=True).squeeze(-1)

        edge_cls_feature = edge_features[:, :self.head_conv, ...]
        edge_reg_feature = edge_features[:, self.head_conv:, ...]
        edge_cls_output = self.edge_cls_convs(edge_cls_feature)
        edge_reg_output = self.edge_reg_convs(edge_reg_feature)

        for k in range(bs):
            edge_indice_k = edge_indices[k, :edge_lens[k]]
            out_features[0][k, :, edge_indice_k[:, 1],
                            edge_indice_k[:, 0]] += edge_cls_output[
                                k, :, :edge_lens[k]]
            out_features[1][k, :, edge_indice_k[:, 1],
                            edge_indice_k[:, 0]] += edge_reg_output[
                                k, :, :edge_lens[k]]

        return out_features
