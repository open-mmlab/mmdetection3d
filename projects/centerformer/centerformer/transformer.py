# modify from https://github.com/TuSimple/centerformer/blob/master/det3d/models/utils/transformer.py # noqa

import torch
from einops import rearrange
from mmcv.cnn.bricks.activation import GELU
from torch import einsum, nn

from .multi_scale_deform_attn import MSDeformAttn


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 n_heads=8,
                 dim_single_head=64,
                 dropout=0.0,
                 out_attention=False):
        super().__init__()
        inner_dim = dim_single_head * n_heads
        project_out = not (n_heads == 1 and dim_single_head == dim)

        self.n_heads = n_heads
        self.scale = dim_single_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity())

    def forward(self, x):
        _, _, _, h = *x.shape, self.n_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class DeformableCrossAttention(nn.Module):

    def __init__(
        self,
        dim_model=256,
        dim_single_head=64,
        dropout=0.3,
        n_levels=3,
        n_heads=6,
        n_points=9,
        out_sample_loc=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            dim_model,
            dim_single_head,
            n_levels,
            n_heads,
            n_points,
            out_sample_loc=out_sample_loc)
        self.dropout = nn.Dropout(dropout)
        self.out_sample_loc = out_sample_loc

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        src,
        query_pos=None,
        reference_points=None,
        src_spatial_shapes=None,
        level_start_index=None,
        src_padding_mask=None,
    ):
        # cross attention
        tgt2, sampling_locations = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.dropout(tgt2)

        if self.out_sample_loc:
            return tgt, sampling_locations
        else:
            return tgt


class DeformableTransformerDecoder(nn.Module):
    """Deformable transformer decoder.

    Note that the ``DeformableDetrTransformerDecoder`` in MMDet has different
    interfaces in multi-head-attention which is customized here. For example,
    'embed_dims' is not a position argument in our customized multi-head-self-
    attention, but is required in MMDet. Thus, we can not directly use the
    ``DeformableDetrTransformerDecoder`` in MMDET.
    """

    def __init__(
        self,
        dim,
        n_levels=3,
        depth=2,
        n_heads=4,
        dim_single_head=32,
        dim_ffn=256,
        dropout=0.0,
        out_attention=False,
        n_points=9,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.n_levels = n_levels
        self.n_points = n_points

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        SelfAttention(
                            dim,
                            n_heads=n_heads,
                            dim_single_head=dim_single_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                        ),
                    ),
                    PreNorm(
                        dim,
                        DeformableCrossAttention(
                            dim,
                            dim_single_head,
                            n_levels=n_levels,
                            n_heads=n_heads,
                            dropout=dropout,
                            n_points=n_points,
                            out_sample_loc=self.out_attention,
                        ),
                    ),
                    PreNorm(dim, FFN(dim, dim_ffn, dropout=dropout)),
                ]))

    def forward(self, x, pos_embedding, src, src_spatial_shapes,
                level_start_index, center_pos):
        if self.out_attention:
            out_cross_attention_list = []
        if pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos)
        reference_points = center_pos[:, :,
                                      None, :].repeat(1, 1, self.n_levels, 1)
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                if center_pos_embedding is not None:
                    x_att, self_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att, self_att = self_attn(x)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                out_cross_attention_list.append(cross_att)
            else:
                if center_pos_embedding is not None:
                    x_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att = self_attn(x)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )

            x = x_att + x
            x = ff(x) + x

        out_dict = {'ct_feat': x}
        if self.out_attention:
            out_dict.update({
                'out_attention':
                torch.stack(out_cross_attention_list, dim=2)
            })
        return out_dict
