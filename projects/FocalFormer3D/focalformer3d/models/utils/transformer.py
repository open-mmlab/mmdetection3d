# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import torch.nn as nn
# --- mmcv.cnn (build_activation_layer, build_norm_layer still valid) ---
from mmcv.cnn import build_activation_layer, build_norm_layer

# --- xavier_init: mmengine.model (new) with mmcv.cnn fallback ---


# ===================================================================
# MultiheadAttention — wrapper around nn.MultiheadAttention
# ===================================================================
class MultiheadAttention(nn.Module):
    """A wrapper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        dropout (float): Dropout on attn_output_weights. Default 0.0.
    """

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, \
            f'embed_dims must be divisible by num_heads. ' \
            f'got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None):
        """Forward function for `MultiheadAttention`.

        Args:
            x (Tensor): Input query [num_query, bs, embed_dims].
            key (Tensor): Key tensor [num_key, bs, embed_dims]. Default None.
            value (Tensor): Value tensor, same shape as key. Default None.
            residual (Tensor): Residual tensor, same shape as x. Default None.
            query_pos (Tensor): Positional encoding for query. Default None.
            key_pos (Tensor): Positional encoding for key. Default None.
            attn_mask (Tensor): Attention mask [num_query, num_key].
            key_padding_mask (Tensor): Key padding mask [bs, num_key].

        Returns:
            Tensor: Output with shape [num_query, bs, embed_dims].
        """
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = x
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query,
            key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        return residual + self.dropout(out)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'dropout={self.dropout})'
        return repr_str


# ===================================================================
# FFN — Feed-Forward Network with residual connection
# ===================================================================
class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): Number of fully-connected layers. Defaults to 2.
        act_cfg (dict): Activation config. Default ReLU.
        dropout (float): Dropout probability. Default 0.0.
        add_residual (bool): Add residual connection. Defaults to True.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, \
            f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    # NOTE: mmcv.cnn.Linear → nn.Linear (identical behavior)
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'add_residual={self.add_residual})'
        return repr_str


# ===================================================================
# TransformerDecoderLayer — single DETR decoder layer
# ===================================================================
class TransformerDecoderLayer(nn.Module):
    """Implements one decoder layer in DETR transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): Hidden dimension of FFNs.
        dropout (float): Dropout probability. Default 0.0.
        order (tuple[str]): The order for decoder layer.
        act_cfg (dict): Activation config. Default ReLU.
        norm_cfg (dict): Normalization config. Default LayerNorm.
        num_fcs (int): Number of fully-connected layers in FFNs.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'multiheadattn', 'norm', 'ffn',
                        'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2):
        super(TransformerDecoderLayer, self).__init__()
        assert isinstance(order, tuple) and len(order) == 6
        assert set(order) == set(['selfattn', 'norm', 'multiheadattn', 'ffn'])
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.num_fcs = num_fcs
        self.pre_norm = order[0] == 'norm'

        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout)
        self.multihead_attn = MultiheadAttention(embed_dims, num_heads,
                                                 dropout)
        self.ffn = FFN(embed_dims, feedforward_channels, num_fcs, act_cfg,
                       dropout)
        self.norms = nn.ModuleList()
        # 3 norm layers in official DETR's TransformerDecoderLayer
        for _ in range(3):
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

    def forward(self,
                x,
                memory,
                memory_pos=None,
                query_pos=None,
                memory_attn_mask=None,
                target_attn_mask=None,
                memory_key_padding_mask=None,
                target_key_padding_mask=None):
        """Forward function for `TransformerDecoderLayer`.

        Args:
            x (Tensor): Input query [num_query, bs, embed_dims].
            memory (Tensor): Encoder output [num_key, bs, embed_dims].
            memory_pos (Tensor): Positional encoding for memory.
            query_pos (Tensor): Positional encoding for query.
            memory_attn_mask (Tensor): Attention mask for memory.
            target_attn_mask (Tensor): Attention mask for x.
            memory_key_padding_mask (Tensor): Key padding mask for memory.
            target_key_padding_mask (Tensor): Key padding mask for x.

        Returns:
            Tensor: Output with shape [num_query, bs, embed_dims].
        """
        norm_cnt = 0
        inp_residual = x

        for layer in self.order:
            if layer == 'selfattn':
                query = key = value = x
                x = self.self_attn(
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos,
                    key_pos=query_pos,
                    attn_mask=target_attn_mask,
                    key_padding_mask=target_key_padding_mask)
                inp_residual = x
            elif layer == 'norm':
                x = self.norms[norm_cnt](x)
                norm_cnt += 1
            elif layer == 'multiheadattn':
                query = x
                key = value = memory
                x = self.multihead_attn(
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos,
                    key_pos=memory_pos,
                    attn_mask=memory_attn_mask,
                    key_padding_mask=memory_key_padding_mask)
                inp_residual = x
            elif layer == 'ffn':
                x = self.ffn(x, inp_residual if self.pre_norm else None)

        return x

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'order={self.order}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'norm_cfg={self.norm_cfg}, '
        repr_str += f'num_fcs={self.num_fcs})'
        return repr_str
