# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """The MLP module used in NerfDet.

    Args:
        input_dim (int): The number of input tensor channels.
        output_dim (int): The number of output tensor channels.
        net_depth (int): The depth of the MLP. Defaults to 8.
        net_width (int): The width of the MLP. Defaults to 256.
        skip_layer (int): The layer to add skip layers to. Defaults to 4.

        hidden_init (Callable): The initialize method of the hidden layers.
        hidden_activation (Callable): The activation function of hidden
            layers, defaults to ReLU.
        output_enabled (bool): If true, the output layers will be used.
            Defaults to True.
        output_init (Optional): The initialize method of the output layer.
        output_activation(Optional): The activation function of output layers.
        bias_enabled (Bool): If true, the bias will be used.
        bias_init (Callable): The initialize method of the bias.
            Defaults to True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layer: int = 4,
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled))
            if (self.skip_layer is not None) and (i % self.skip_layer
                                                  == 0) and (i > 0):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled)
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):

        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (self.skip_layer is not None) and (i % self.skip_layer
                                                  == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    """The Nerf-MLP Module.

    Args:
        input_dim (int): The number of input tensor channels.
        condition_dim (int): The number of condition tensor channels.
        feature_dim (int): The number of feature channels. Defaults to 0.
        net_depth (int): The depth of the MLP. Defaults to 8.
        net_width (int): The width of the MLP. Defaults to 256.
        skip_layer (int): The layer to add skip layers to. Defaults to 4.
        net_depth_condition (int): The depth of the second part of MLP.
            Defaults to 1.
        net_width_condition (int): The width of the second part of MLP.
            Defaults to 128.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        feature_dim: int = 0,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layer: int = 4,
        net_depth_condition: int = 1,
        net_width_condition: int = 128,
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim + feature_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x, features=None):
        """Calculate the raw sigma."""
        if features is not None:
            x = self.base(torch.cat([x, features], dim=-1))
        else:
            x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None, features=None):
        if features is not None:
            x = self.base(torch.cat([x, features], dim=-1))
        else:
            x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) +
                    [n_dim]).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusodial Positional Encoder used in NeRF."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            'scales', torch.tensor([2**i for i in range(min_deg, max_deg)]))

    @property
    def latent_dim(self) -> int:
        return (int(self.use_identity) +
                (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRF(nn.Module):
    """The Nerf-MLP with the positional encoder.

    Args:
        net_depth (int): The depth of the MLP. Defaults to 8.
        net_width (int): The width of the MLP. Defaults to 256.
        skip_layer (int): The layer to add skip layers to. Defaults to 4.
        feature_dim (int): The number of feature channels. Defaults to 0.
        net_depth_condition (int): The depth of the second part of MLP.
            Defaults to 1.
        net_width_condition (int): The width of the second part of MLP.
            Defaults to 128.
    """

    def __init__(self,
                 net_depth: int = 8,
                 net_width: int = 256,
                 skip_layer: int = 4,
                 feature_dim: int = 0,
                 net_depth_condition: int = 1,
                 net_width_condition: int = 128):
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            feature_dim=feature_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def query_density(self, x, features=None):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x, features)
        return F.relu(sigma)

    def forward(self, x, condition=None, features=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition, features=features)
        return torch.sigmoid(rgb), F.relu(sigma)
