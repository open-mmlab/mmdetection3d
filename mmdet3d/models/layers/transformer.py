# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.registry import MODELS
from torch import Tensor
from torch import nn as nn

from mmdet3d.utils import ConfigType, OptMultiConfig


@MODELS.register_module()
class GroupFree3DMHA(MultiheadAttention):
    """A wrapper for torch.nn.MultiheadAttention for GroupFree3D.

    This module implements MultiheadAttention with identity connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Defaults to 0.0.
        proj_drop (float): A Dropout layer. Defaults to 0.0.
        dropout_layer (ConfigType): The dropout_layer used when adding
            the shortcut. Defaults to dict(type='DropOut', drop_prob=0.).
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim).
            Defaults to False.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 dropout_layer: ConfigType = dict(
                     type='DropOut', drop_prob=0.),
                 init_cfg: OptMultiConfig = None,
                 batch_first: bool = False,
                 **kwargs) -> None:
        super(GroupFree3DMHA,
              self).__init__(embed_dims, num_heads, attn_drop, proj_drop,
                             dropout_layer, init_cfg, batch_first, **kwargs)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                identity: Tensor,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """Forward function for `GroupFree3DMHA`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                If None, the ``query`` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link. If None, `x` will be used.
            query_pos (Tensor, optional): The positional encoding for query,
                with the same shape as `x`. Defaults to None.
                If not None, it will be added to `x` before forward function.
            key_pos (Tensor, optional): The positional encoding for `key`,
                with the same shape as `key`. Defaults to None. If not None,
                it will be added to `key` before forward function. If None,
                and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor, optional): ByteTensor mask with shape
                [num_queries, num_keys].
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
            key_padding_mask (Tensor, optional): ByteTensor with shape
                [bs, num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.

        Returns:
            Tensor: Forwarded results with shape [num_queries, bs, embed_dims].
        """

        if hasattr(self, 'operation_name'):
            if self.operation_name == 'self_attn':
                value = value + query_pos
            elif self.operation_name == 'cross_attn':
                value = value + key_pos
            else:
                raise NotImplementedError(
                    f'{self.__class__.name} '
                    f"can't be used as {self.operation_name}")
        else:
            value = value + query_pos

        return super(GroupFree3DMHA, self).forward(
            query=query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)


@MODELS.register_module()
class ConvBNPositionalEncoding(nn.Module):
    """Absolute position embedding with Conv learning.

    Args:
        input_channel (int): Input features dim.
        num_pos_feats (int): Output position features dim.
            Defaults to 288 to be consistent with seed features dim.
    """

    def __init__(self, input_channel: int, num_pos_feats: int = 288) -> None:
        super(ConvBNPositionalEncoding, self).__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz: Tensor) -> Tensor:
        """Forward pass.

        Args:
            xyz (Tensor): (B, N, 3) The coordinates to embed.

        Returns:
            Tensor: (B, num_pos_feats, N) The embedded position features.
        """
        xyz = xyz.permute(0, 2, 1)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
