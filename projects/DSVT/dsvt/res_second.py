# modified from https://github.com/Haiyang-W/DSVT
from typing import Sequence, Tuple

from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import OptMultiConfig


class BasicResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01))
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


@MODELS.register_module()
class ResSECOND(BaseModule):
    """Backbone network for DSVT. The difference between `ResSECOND` and
    `SECOND` is that the basic block in this module contains residual layers.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        blocks_nums (list[int]): Number of blocks in each stage.
        layer_strides (list[int]): Strides of each stage.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 128,
                 out_channels: Sequence[int] = [128, 128, 256],
                 blocks_nums: Sequence[int] = [1, 2, 2],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 init_cfg: OptMultiConfig = None) -> None:
        super(ResSECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(blocks_nums)
        assert len(out_channels) == len(blocks_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, block_num in enumerate(blocks_nums):
            cur_layers = [
                BasicResBlock(
                    in_filters[i],
                    out_channels[i],
                    stride=layer_strides[i],
                    downsample=True)
            ]
            for _ in range(block_num):
                cur_layers.append(
                    BasicResBlock(out_channels[i], out_channels[i]))
            blocks.append(nn.Sequential(*cur_layers))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
