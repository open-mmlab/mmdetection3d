from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import MultiheadAttention


@ATTENTION.register_module()
class GroupFree3DMultiheadAttention(MultiheadAttention):
    """A warpper for torch.nn.MultiheadAttention for GroupFree3D.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        attn_drop (float): A Dropout layer on attn_output_weights. Default 0.0.
        proj_drop (float): A Dropout layer. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropOut', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop,
                         dropout_layer, init_cfg, batch_first, **kwargs)

    def forward(self,
                query,
                key,
                value,
                residual,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `GroupFree3DMultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            residual (Tensor): This tensor, with the same shape as x,
                will be used for the residual link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Same in `nn.MultiheadAttention.forward`. Defaults to None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
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

        return super(GroupFree3DMultiheadAttention, self).forward(
            query=query,
            key=key,
            value=value,
            residual=residual,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
