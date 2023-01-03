# modify from https://github.com/TuSimple/centerformer/blob/master/det3d/models/ops/modules/ms_deform_attn.py # noqa

import math
from typing import Optional

import torch
import torch.nn.functional as F
from mmcv.utils import ext_loader
from torch import Tensor, nn
from torch.autograd.function import Function, once_differentiable
from torch.nn.init import constant_, xavier_uniform_

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        """GPU/MLU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.
        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU/MLU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.
        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module. Note that the difference
    between this implementation and the implementation in MMCV is that the
    dimension of input and hidden embedding in the multi-attention-head can be
    specified respectively.

    Args:
        dim_model (int, optional): The input and output dimension in the model.
            Defaults to 256.
        dim_single_head (int, optional): hidden dimension in the single head.
            Defaults to 64.
        n_levels (int, optional): number of feature levels. Defaults to 4.
        n_heads (int, optional): number of attention heads. Defaults to 8.
        n_points (int, optional): number of sampling points per attention head
            per feature level. Defaults to 4.
        out_sample_loc (bool, optional): Whether to return the sampling
            location. Defaults to False.
    """

    def __init__(self,
                 dim_model=256,
                 dim_single_head=64,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 out_sample_loc=False):
        super().__init__()

        self.im2col_step = 64

        self.dim_model = dim_model
        self.dim_single_head = dim_single_head
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.out_sample_loc = out_sample_loc

        self.sampling_offsets = nn.Linear(dim_model,
                                          n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(dim_model,
                                           n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(dim_model, dim_single_head * n_heads)
        self.output_proj = nn.Linear(dim_single_head * n_heads, dim_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                       self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query: Tensor,
                reference_points: Tensor,
                input_flatten: Tensor,
                input_spatial_shapes: Tensor,
                input_level_start_index: Tensor,
                input_padding_mask: Optional[Tensor] = None):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): (N, num_query, C)
            reference_points (Tensor): (N, num_query, n_levels, 2). The
                normalized reference points with shape
                (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            input_flatten (Tensor): _description_
            input_spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            input_level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            input_padding_mask (Optional[Tensor], optional): The padding mask
                for value. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: forwarded results.
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.dim_single_head)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights,
                                      -1).view(N, Len_q, self.n_heads,
                                               self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1).to(sampling_offsets)

            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]  # noqa: E501
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5   # noqa: E501
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'  # noqa: E501
                .format(reference_points.shape[-1]))
        output = MultiScaleDeformableAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        if self.out_sample_loc:
            return output, torch.cat(
                (sampling_locations, attention_weights[:, :, :, :, :, None]),
                dim=-1)
        else:
            return output, None
