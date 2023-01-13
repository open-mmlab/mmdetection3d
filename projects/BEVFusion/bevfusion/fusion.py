import random
from typing import List

import torch
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class AddFuser(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.transforms = nn.ModuleList()
        for k in range(len(in_channels)):
            self.transforms.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels[k], out_channels, 3, padding=1,
                        bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        features = []
        for transform, input in zip(self.transforms, inputs):
            features.append(transform(input))

        weights = [1] * len(inputs)
        if self.training and random.random() < self.dropout:
            index = random.randint(0, len(inputs) - 1)
            weights[index] = 0

        return sum(w * f for w, f in zip(weights, features)) / sum(weights)
