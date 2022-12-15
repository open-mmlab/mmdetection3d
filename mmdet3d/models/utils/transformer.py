# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from einops import rearrange
from torch import einsum, nn
from torch.nn import functional as F

from .multi_scale_deform_attn import MSDeformAttn


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# transformer layer
class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_CA(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)


class FeedForward(nn.Module):

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


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 out_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity())

    def forward(self, x):
        _, _, _, h = *x.shape, self.heads
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


class Cross_attention(nn.Module):

    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 out_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity())

    def forward(self, x, y):
        b, _, _, _, h = *y.shape, self.heads
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> (b n) h 1 d', h=h)
        k, v = map(lambda t: rearrange(t, 'b n m (h d) -> (b n) h m d', h=h),
                   kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b n) h 1 d -> b n (h d)', b=b)

        if self.out_attention:
            return self.to_out(out), rearrange(
                attn, '(b n) h i j -> b n h (i j)', b=b)
        else:
            return self.to_out(out)


class DeformableTransformerCrossAttention(nn.Module):

    def __init__(
        self,
        d_model=256,
        d_head=64,
        dropout=0.3,
        n_levels=3,
        n_heads=6,
        n_points=9,
        out_sample_loc=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            d_model,
            d_head,
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


class Transformer(nn.Module):

    def __init__(
        self,
        dim,
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=256,
        dropout=0.0,
        out_attention=False,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                        ),
                    ),
                    PreNorm_CA(
                        dim,
                        Cross_attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                        ),
                    ),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

    def forward(self,
                x,
                pos_embedding=None,
                center_pos=None,
                y=None,
                neighbor_pos=None):
        if self.out_attention:
            out_cross_attention_list = []
        if center_pos is not None and pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos)
        if neighbor_pos is not None and pos_embedding is not None:
            neighbor_pos_embedding = pos_embedding(neighbor_pos)
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                if pos_embedding is not None:
                    x_att, self_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att, cross_att = cross_attn(x + center_pos_embedding,
                                                  y + neighbor_pos_embedding)
                else:
                    x_att, self_att = self_attn(x)
                    x = x_att + x
                    x_att, cross_att = cross_attn(x, y)
                out_cross_attention_list.append(cross_att)
            else:
                if pos_embedding is not None:
                    x_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att = cross_attn(x + center_pos_embedding,
                                       y + neighbor_pos_embedding)
                else:
                    x_att = self_attn(x)
                    x = x_att + x
                    x_att = cross_attn(x, y)

            x = x_att + x
            x = ff(x) + x

        out_dict = {'ct_feat': x}
        if self.out_attention:
            out_dict.update({
                'out_attention':
                torch.stack(out_cross_attention_list, dim=2)
            })
        return out_dict


class Deform_Transformer(nn.Module):

    def __init__(
        self,
        dim,
        levels=3,
        depth=2,
        heads=4,
        dim_head=32,
        mlp_dim=256,
        dropout=0.0,
        out_attention=False,
        n_points=9,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.levels = levels
        self.n_points = n_points

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                        ),
                    ),
                    PreNorm_CA(
                        dim,
                        DeformableTransformerCrossAttention(
                            dim,
                            dim_head,
                            n_levels=levels,
                            n_heads=heads,
                            dropout=dropout,
                            n_points=n_points,
                            out_sample_loc=self.out_attention,
                        ),
                    ),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

    def forward(self, x, pos_embedding, src, src_spatial_shapes,
                level_start_index, center_pos):
        if self.out_attention:
            out_cross_attention_list = []
        if pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos)
        reference_points = center_pos[:, :,
                                      None, :].repeat(1, 1, self.levels, 1)
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
