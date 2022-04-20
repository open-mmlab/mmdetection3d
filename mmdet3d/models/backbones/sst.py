# Copyright (c) OpenMMLab. All rights reserved.
from collections import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule
from torch.utils.checkpoint import checkpoint

from mmdet3d import digit_version
from mmdet3d.ops.sst_modules import seq_to_win, win_to_seq
from mmdet.models import BACKBONES


class SparseRegionAttention(BaseModule):
    """Do the Region Attention.

    The code is modified from original implementation
    https://github.com/TuSimple/SST

    Args:
        embed_dim (int): Number of dimension of feature embedding.
        num_heads (int): Number of heads in attention
        dropout (float): Drop probability when do the attention
        init_cfg (:obj:`ConfigDict`): Initialization config.
            Defaults to None.
    """

    def __init__(self, embed_dim, num_heads, dropout, init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout)

    def forward(
        self,
        feat,
        batching_position_embed,
        batching_padding_mask,
        seq_win_mappings,
    ):
        """
        Args:
            feat (tensor): The sequence features to be transformed.
                has shape (N, FEAT_DIM).
            batching_position_embed (list[tensor]): The list indicate
                the different batching. Each tensor is the position
                embedding of corresponding batching.
                Each tensor has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
            batching_padding_mask (list[tensor]): The  list
                indicate the different batching. The tensor is
                the padding mask of corresponding batching, each
                tensor has shape (NUM_WINS, NUM_TOKEN).
            seq_win_mappings (list[tuple]): Each tuple contains
            three important information of corresponding
            batching.

                - A identical index of voxel in corresponding
                  batching windows
                - The index of voxel in original sequence.
                - The number of token of each window in this batching.

        Returns:
            tensor: The sequence features after attention.
            Has shape (N, FEAT_DIM).
        """
        batching_feat_list = seq_to_win(feat, seq_win_mappings)
        attn_feat_list = []
        for single_feat, single_position_embed, single_padding_mask \
                in zip(batching_feat_list,
                       batching_position_embed,
                       batching_padding_mask):

            single_feat = single_feat.permute(1, 0, 2)
            v = single_feat

            if single_position_embed is not None:
                single_position_embed = single_position_embed.permute(1, 0, 2)
                q = k = single_feat + single_position_embed
            else:
                q = k = single_feat

            attn_feat, _ = self.attn(
                q, k, value=v, key_padding_mask=single_padding_mask)
            attn_feat_list.append(attn_feat.permute(1, 0, 2))

        feat = win_to_seq(attn_feat_list, seq_win_mappings)

        return feat


class SSTLayer(BaseModule):
    """Single transformer layer of SST.

    Args:
        embed_dims(int): Number of dimension of feature embedding.
        num_heads (int): Number of heads in attention
        feedforward_channels (int): Number of channels in FFN.
        attn_dropout(float): Drop probability when do the attention:
        ffn_dropout(float): Drop probability in the FFN:
        norm_cfg (:obj:`ConfigDict`): Config for normalization layer.
        act_cfg (:obj:`ConfigDict`): Config for activation layer.
        post_norm (bool): Whether use post norm. Defaults to True.
        init_cfg (:obj:`ConfigDict`): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels=256,
                 attn_dropout=0.0,
                 ffn_dropout=0.0,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 post_norm=True,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.attn = SparseRegionAttention(embed_dims, num_heads, attn_dropout)

        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.dropout1 = nn.Dropout(ffn_dropout)
        self.dropout2 = nn.Dropout(ffn_dropout)

        self.activation = build_activation_layer(act_cfg)
        self.post_norm = post_norm
        self.fp16_enabled = False

    @auto_fp16(apply_to=('feat'))
    def forward(
        self,
        feat,
        batching_position_embed,
        batching_padding_mask,
        seq_win_mappings,
    ):
        """
        Args:
            feat (tensor): The sequence features to be transformed.
                has shape (N, FEAT_DIM).
            batching_position_embed (list[tensor]): The list indicate
                the different batching. Each tensor is the position
                embedding of corresponding batching.
                Each tensor has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
            batching_padding_mask (list[tensor]): The  list
                indicate the different batching. The tensor is
                the padding mask of corresponding batching, each
                tensor has shape (NUM_WINS, NUM_TOKEN).
            seq_win_mappings (list[tuple]): Each tuple contains
            three important information of corresponding
            batching.

                - A identical index of voxel in corresponding
                  batching windows
                - The index of voxel in original sequence.
                - The number of token of each window in this batching.

        Returns:
            tensor: The sequence features after attention.
            Has shape (N, FEAT_DIM).
        """
        if self.post_norm:
            attn_feat = self.attn(feat, batching_position_embed,
                                  batching_padding_mask, seq_win_mappings)
            feat = feat + self.dropout1(attn_feat)
            feat = self.norm1(feat)
            ffn_feat = self.linear2(
                self.dropout(self.activation(self.linear1(feat))))
            feat = feat + self.dropout2(ffn_feat)
            feat = self.norm2(feat)
        else:
            feat = self.norm1(feat)
            attn_feat = self.attn(feat, batching_position_embed,
                                  batching_padding_mask, seq_win_mappings)
            feat = feat + self.dropout1(attn_feat)
            feat = self.norm2(feat)
            ffn_feat = self.linear2(
                self.dropout(self.activation(self.linear1(feat))))
            feat = feat + self.dropout2(ffn_feat)
        return feat


class SSTLayerSequence(BaseModule):
    """Sequence of SST layer.

    Args:
        layer_cfg (:obj:`ConfigDict`): Config for single SST layer.
            It has following keys:

            - embed_dims(int): Number of dimension of feature embedding.
            - num_heads (int): Number of heads in attention
            - feedforward_channels (int): Number of channels in FFN.
            - attn_dropout(float): Drop probability when do the attention:
            - ffn_dropout(float): Drop probability in the FFN:
            - norm_cfg (:obj:`ConfigDict`): Config for normalization layer.
            - act_cfg (:obj:`ConfigDict`): Config for activation layer.
            - post_norm (bool): Whether use post norm. Defaults to True.

        init_cfg (:obj:`ConfigDict`): Initialization config.
            Defaults to None.
    """

    def __init__(self, layer_cfg, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        sst_layer1 = SSTLayer(**layer_cfg)
        sst_layer2 = SSTLayer(**layer_cfg)
        self.encoder_layers = nn.ModuleList([sst_layer1, sst_layer2])

    def forward(
        self,
        feat,
        batching_position_embed_list,
        batching_padding_mask_list,
        seq_win_mapping_list,
        using_checkpoint=False,
    ):
        """
        Args:
            feat (tensor): Voxel features in shape (N, C).
            batching_position_embed_list (list[list[tensor]]):
                The outer list indicate the attentions. The
                inner list indicate the different batching.
                Each tensor is the position
                embedding of corresponding batching.
                Each tensor has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
            batching_padding_mask_list (list[list[tensor]]):
                The outer list indicate the attentions.The inner list
                indicate the different batching. The tensor is
                the padding mask of corresponding batching, each
                tensor has shape (NUM_WINS, NUM_TOKEN).
            seq_win_mapping_list: (list[list[tuple]]): The outer
                list indicate the attentions.The inner list indicate
                the different batching. Each tuple contains 3 important
                information of corresponding batching.

                - A identical index of voxel in corresponding
                  batching windows
                - The index of voxel in original sequence.
                - The number of token of each window in this batching.

        Returns:
            tensor: Voxel features in shape (N, C).
        """
        num_attns = len(batching_position_embed_list)
        for attn_index in range(num_attns):
            batching_position_embed = batching_position_embed_list[attn_index]
            seq_win_mapping = seq_win_mapping_list[attn_index]
            batching_padding_mask = batching_padding_mask_list[attn_index]

            layer = self.encoder_layers[attn_index]
            if using_checkpoint and self.training:
                feat = checkpoint(layer, feat, batching_position_embed,
                                  batching_padding_mask, seq_win_mapping)
            else:
                feat = layer(feat, batching_position_embed,
                             batching_padding_mask, seq_win_mapping)

        return feat


@BACKBONES.register_module()
class SST(nn.Module):
    """Single-stride Sparse Transformer.

    The code is modified from original implementation
    https://github.com/TuSimple/SST

    Args:
        in_channel (int, optional): The number of channels in first
            lateral_linear. Skip this linear when `in_channel` is
            None. Defaults to None.
        layer_cfg (:obj:`ConfigDict`): Config for single SST layer.
            It has following keys:

            - embed_dims(int): Number of dimension of feature embedding.
            - num_heads (int): Number of heads in attention
            - feedforward_channels (int): Number of channels in FFN.
            - attn_dropout(float): Drop probability when do the attention:
            - ffn_dropout(float): Drop probability in the FFN:
            - norm_cfg (:obj:`ConfigDict`): Config for normalization layer.
            - act_cfg (:obj:`ConfigDict`): Config for activation layer.
            - post_norm (bool): Whether use post norm. Defaults to True.

        num_layers (int): Number of :obj:`SSTLayerSequences`. Defaults
            to 6.
        output_shape (tuple[int]): Shape of output bev
            feature, arranged as (H, W). Defaults to [468, 468].
        num_bev_conv (dict): Config for `bev_conv`. Defaults to
            dict(type='Conv2d').
        bev_conv_args (list[dict]): The detail arguments for each
            bev convolution.
        bev_norm_cfg (dict): Config for the normalization layer.
            Defaults to
            dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)
        checkpoint_blocks (list[int], optional): Block IDs
            (0 to num_blocks - 1) to use checkpoint. Defaults
            to None.
    """

    def __init__(self,
                 in_channel=None,
                 layer_cfg=dict(
                     embed_dims=128,
                     num_heads=8,
                     feedforward_channels=256,
                     attn_dropout=0.0,
                     ffn_dropout=0.0,
                     norm_cfg=dict(type='LN'),
                     act_cfg=dict(type='GELU'),
                     post_norm=True),
                 num_layers=6,
                 output_shape=[468, 468],
                 bev_conv_cfg=dict(type='Conv2d'),
                 num_bev_convs=3,
                 bev_conv_args=[
                     dict(kernel_size=3, dilation=1, padding=1, stride=1),
                     dict(kernel_size=3, dilation=1, padding=1, stride=1),
                     dict(kernel_size=3, dilation=2, padding=2, stride=1),
                 ],
                 bev_norm_cfg=dict(
                     type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
                 checkpoint_blocks=None,
                 **kwargs):
        super().__init__()
        embed_dims = layer_cfg.get('embed_dims', 128)
        if checkpoint_blocks is None:
            self.checkpoint_blocks = []
        else:
            assert isinstance(checkpoint_blocks, Sequence)
            for item in checkpoint_blocks:
                assert isinstance(item, int)
            self.checkpoint_blocks = checkpoint_blocks

        if self.checkpoint_blocks:
            from torch import __version__
            assert (digit_version(__version__) >= digit_version(1.9)), (
                'In PyTorch <= 1.9, '
                'checkpoint function seems not able to '
                'receive dict as parameters. '
                'Better to use PyTorch >= 1.9.')
        if in_channel is not None:
            self.lateral_linear = nn.Linear(in_channel, embed_dims)
        else:
            self.lateral_linear = None

        # Sparse Regional Attention Blocks
        layer_sequence_list = []
        for i in range(num_layers):
            layer_sequence_list.append(SSTLayerSequence(layer_cfg))
        self.encoder = nn.ModuleList(layer_sequence_list)
        self.output_shape = output_shape
        self.num_bev_convs = num_bev_convs

        if self.num_bev_convs > 0:
            if isinstance(bev_conv_args, dict):
                bev_conv_args = [bev_conv_args]
            assert len(bev_conv_args) == self.num_bev_convs
            conv_list = []
            for bev_conv_arg in bev_conv_args:
                bev_conv = ConvModule(
                    conv_cfg=bev_conv_cfg,
                    norm_cfg=bev_norm_cfg,
                    **bev_conv_arg,
                )
                conv_list.append(bev_conv)

            self.bev_conv_layers = nn.ModuleList(conv_list)
        else:
            self.bev_conv_layers = None

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): Contains 5 part

                - voxel_feats (tensor): Voxel features in shape (N, C).
                - voxel_coors (tensor): Coordinates in shape (N, 4),
                  the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
                - batching_position_embed_list (list[list[tensor]]):
                  The outer list indicate the attentions. The inner list
                  indicate the different batching. Each tensor is the position
                  embedding of corresponding batching.
                  Each tensor has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
                - batching_padding_mask_list (list[list[tensor]]):
                  The outer list indicate the attentions.The inner list
                  indicate the different batching. The tensor is
                  the padding mask of corresponding batching, each
                  tensor has shape (NUM_WINS, NUM_TOKEN).
                - seq_win_mapping_list: (list[list[tuple]]): The outer
                  list indicate the attentions.The inner list indicate
                  the different batching. Each tuple contains 3 important
                  information of corresponding batching.

                    - A identical index of voxel in corresponding
                      batching windows
                    - The index of voxel in original sequence.
                    - The number of token of each window in this batching.

        Returns:
            list(tensor): Bev format feature. Has shape (B, C, out_h, out_w)
        """
        (voxel_feats, voxel_coors, batching_position_embed_list,
         batching_padding_mask_list, seq_win_mapping_list) = inputs
        batch_size = voxel_coors[:, 0].max().item() + 1
        batch_size = int(batch_size)
        if self.lateral_linear:
            voxel_feats = self.lateral_linear(voxel_feats)
        for i, sst_layer_sequence in enumerate(self.encoder):
            voxel_feats = sst_layer_sequence(
                voxel_feats,
                batching_position_embed_list,
                batching_padding_mask_list,
                seq_win_mapping_list,
                using_checkpoint=i in self.checkpoint_blocks)

        output = self.to_bev_feats(voxel_feats, voxel_coors, batch_size)

        if self.bev_conv_layers:
            for conv_module in self.bev_conv_layers:
                output = conv_module(output)

        output_list = []
        output_list.append(output)

        return output_list

    def init_weights(self):
        """Initialize the weights of models."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_bev_feats(self, voxel_feats, voxel_coors, batch_size):
        """Convert feature to the BEV format.

        Args:
            voxel_feats (tensor): Voxel features in shape (N, C).
            voxel_coors (tensor): Coordinates in shape (N, 4),
                arrange as (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size of features.

        Returns:
            tensor: Bev format feature. Has shape (B, C, out_h, out_w)
        """
        out_h, out_w = self.output_shape
        feat_dim = voxel_feats.shape[-1]

        bev_feat_list = []
        for batch_index in range(batch_size):
            bev_feat = voxel_feats.new_zeros(feat_dim, out_w * out_h)
            batch_mask = voxel_coors[:, 0] == batch_index
            temp_voxel_coors = voxel_coors[batch_mask, :]
            indices = temp_voxel_coors[:, 2] * out_w + temp_voxel_coors[:, 3]
            indices = indices.type(torch.long)
            temp_voxel_feats = voxel_feats[batch_mask, :]  # [n, c]
            temp_voxel_feats = temp_voxel_feats.t()  # [c, n]
            bev_feat[:, indices] = temp_voxel_feats
            bev_feat_list.append(bev_feat)

        bev_feats = torch.stack(bev_feat_list, 0)
        bev_feats = bev_feats.view(batch_size, feat_dim, out_h, out_w)

        return bev_feats
