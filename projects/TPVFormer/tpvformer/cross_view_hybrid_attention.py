import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmengine.model import BaseModule, constant_init, xavier_init

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TPVFormerOCCCrossViewHybridAttention(BaseModule):
    """Cross view hybrid attention module used in TPVFormer.
    Based on deformable attention.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 num_tpv_queue=2):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_tpv_queue = num_tpv_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * num_tpv_queue, num_tpv_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * num_tpv_queue,
                                           num_tpv_queue * num_heads * num_levels * num_points)
        self.attention_weights = self.attention_weights.to('cuda')
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = self.value_proj.to('cuda')
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = self.output_proj.to('cuda')
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_tpv_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        if value is None:
            value = torch.cat([query, query], 0)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_tpv_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(self.num_tpv_queue*bs,
                              num_value, self.num_heads, -1)
        self.sampling_offsets = self.sampling_offsets.to('cuda') # gp

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_tpv_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_tpv_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_tpv_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(3, 0, 1, 2, 4, 5)\
            .reshape(bs*self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        # output shape (bs*num_tpv_queue, num_query, embed_dims)
        # (bs*num_tpv_queue, num_query, embed_dims) -> (num_query, embed_dims, bs*num_tpv_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_tpv_queue) -> (num_query, embed_dims, bs)
        output = (output[..., :bs] + output[..., bs:]) / self.num_tpv_queue

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@MODELS.register_module()
class TPVCrossViewHybridAttention(BaseModule):
    """TPVFormer Cross-view Hybrid Attention Module."""

    def __init__(self,
                 tpv_h: int,
                 tpv_w: int,
                 tpv_z: int,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_points: int = 4,
                 num_anchors: int = 2,
                 init_mode: int = 0,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = 3
        self.num_points = num_points
        self.num_anchors = num_anchors
        self.init_mode = init_mode
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
        self.output_proj = nn.ModuleList(
            [nn.Linear(embed_dims, embed_dims) for _ in range(3)])
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * 3 * num_points * 2)
            for _ in range(3)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * 3 * (num_points + 1))
            for _ in range(3)
        ])
        self.value_proj = nn.ModuleList(
            [nn.Linear(embed_dims, embed_dims) for _ in range(3)])

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        device = next(self.parameters()).device
        # self plane
        theta_self = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_self = torch.stack(
            [theta_self.cos(), theta_self.sin()], -1)  # H, 2
        grid_self = grid_self.view(self.num_heads, 1,
                                   2).repeat(1, self.num_points, 1)
        for j in range(self.num_points):
            grid_self[:, j, :] *= (j + 1) / 2

        if self.init_mode == 0:
            # num_phi = 4
            phi = torch.arange(
                4, dtype=torch.float32, device=device) * (2.0 * math.pi / 4)
            assert self.num_heads % 4 == 0
            num_theta = int(self.num_heads / 4)
            theta = torch.arange(
                num_theta, dtype=torch.float32, device=device) * (
                    math.pi / num_theta) + (math.pi / num_theta / 2)  # 3
            x = torch.matmul(theta.sin().unsqueeze(-1),
                             phi.cos().unsqueeze(0)).flatten()
            y = torch.matmul(theta.sin().unsqueeze(-1),
                             phi.sin().unsqueeze(0)).flatten()
            z = theta.cos().unsqueeze(-1).repeat(1, 4).flatten()
            xyz = torch.stack([x, y, z], dim=-1)  # H, 3

        elif self.init_mode == 1:

            xyz = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0],
                   [-1, 0, 0]]
            xyz = torch.tensor(xyz, dtype=torch.float32, device=device)

        grid_hw = xyz[:, [0, 1]]  # H, 2
        grid_zh = xyz[:, [2, 0]]
        grid_wz = xyz[:, [1, 2]]

        for i in range(3):
            grid = torch.stack([grid_hw, grid_zh, grid_wz], dim=1)  # H, 3, 2
            grid = grid.unsqueeze(2).repeat(1, 1, self.num_points, 1)

            grid = grid.reshape(self.num_heads, self.num_levels,
                                self.num_anchors, -1, 2)
            for j in range(self.num_points // self.num_anchors):
                grid[:, :, :, j, :] *= 2 * (j + 1)
            grid = grid.flatten(2, 3)
            grid[:, i, :, :] = grid_self

            constant_init(self.sampling_offsets[i], 0.)
            self.sampling_offsets[i].bias.data = grid.view(-1)

            constant_init(self.attention_weights[i], val=0., bias=0.)
            attn_bias = torch.zeros(
                self.num_heads, 3, self.num_points + 1, device=device)
            attn_bias[:, i, -1] = 10
            self.attention_weights[i].bias.data = attn_bias.flatten()
            xavier_init(self.value_proj[i], distribution='uniform', bias=0.)
            xavier_init(self.output_proj[i], distribution='uniform', bias=0.)

    def get_sampling_offsets_and_attention(
            self, queries: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(
                zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels,
                                       self.num_points, 2)
            offsets.append(offset)

            attention = attn(query).reshape(bs, l, self.num_heads, 3, -1)
            level_attention = attention[:, :, :, :,
                                        -1:].softmax(-2)  # bs, l, H, 3, 1
            attention = attention[:, :, :, :, :-1]
            attention = attention.softmax(-1)  # bs, l, H, 3, p
            attention = attention * level_attention
            attns.append(attention)

        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_output(self, output: Tensor, lens: List[int]) -> List[Tensor]:
        outputs = torch.split(output, [lens[0], lens[1], lens[2]], dim=1)
        return outputs

    def forward(self,
                query: List[Tensor],
                identity: Optional[List[Tensor]] = None,
                query_pos: Optional[List[Tensor]] = None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None):
        identity = query if identity is None else identity
        if query_pos is not None:
            query = [q + p for q, p in zip(query, query_pos)]

        # value proj
        query_lens = [q.shape[1] for q in query]
        value = [layer(q) for layer, q in zip(self.value_proj, query)]
        value = torch.cat(value, dim=1)
        bs, num_value, _ = value.shape
        value = value.view(bs, num_value, self.num_heads, -1)

        # sampling offsets and weights
        sampling_offsets, attention_weights = \
            self.get_sampling_offsets_and_attention(query)

        if reference_points.shape[-1] == 2:
            """For each tpv query, it owns `num_Z_anchors` in 3D space that
            having different heights. After projecting, each tpv query has
            `num_Z_anchors` reference points in each 2D image. For each
            referent point, we sample `num_points` sampling points.

            For `num_Z_anchors` reference points,
            it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, _, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, :, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape  # noqa
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors,
                num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape  # noqa

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        outputs = self.reshape_output(output, query_lens)

        results = []
        for out, layer, drop, residual in zip(outputs, self.output_proj,
                                              self.dropout, identity):
            results.append(residual + drop(layer(out)))

        return results
