# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

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


def gen_sineembed_for_position_all(pos_tensor):
    """Generate sinusoidal positional embeddings for all dimensions.

    Args:
        pos_tensor (torch.Tensor): Position tensor [B, N, D].

    Returns:
        torch.Tensor: Sinusoidal embeddings [B, N, D*128].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000**(2 * (dim_t // 2) / 128)

    x_embed = pos_tensor[:, :, :] * scale
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()],
        dim=-1).flatten(-2)

    return pos_x


def gen_sineembed_for_position(pos_tensor):
    """Generate sinusoidal positional embeddings for 2D or 4D positions.

    Args:
        pos_tensor (torch.Tensor): Position tensor [N, B, 2] or [N, B, 4].
            For 2D: (x, y). For 4D: (x, y, w, h).

    Returns:
        torch.Tensor: Sinusoidal embeddings [N, B, 256] or [N, B, 512].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000**(2 * (dim_t // 2) / 128)

    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                        dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                        dim=3).flatten(2)

    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)

    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                            dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                            dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

    else:
        raise ValueError(
            f'Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}')

    return pos
