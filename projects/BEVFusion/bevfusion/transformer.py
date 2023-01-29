# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DetrTransformerDecoderLayer
from torch import Tensor, nn

from mmdet3d.registry import MODELS


class PositionEncodingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@MODELS.register_module()
class TransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                 pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 **kwargs):
        super().__init__(**kwargs)
        self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        if self.self_posembed is not None and query_pos is not None:
            query_pos = self.self_posembed(query_pos).transpose(1, 2)
        else:
            query_pos = None
        if self.cross_posembed is not None and key_pos is not None:
            key_pos = self.cross_posembed(key_pos).transpose(1, 2)
        else:
            key_pos = None
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        # Note that the `value` (equal to `query`) is encoded with `query_pos`.
        # This is different from the standard DETR Decoder Layer.
        query = self.self_attn(
            query=query,
            key=query,
            value=query + query_pos,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        query = self.cross_attn(
            query=query,
            key=key,
            value=key + key_pos,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        query = query.transpose(1, 2)
        return query
