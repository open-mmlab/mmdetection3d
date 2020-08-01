import numpy as np
import torch
from mmcv.cnn import build_norm_layer, xavier_init
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module
class CenterPointRPN(nn.Module):
    """RPN used in CenterPoint.

    Args:
        layer_nums (list[int]): Number of layers for each block.
        downsample_strides (list[int]): Strides used to
            downsample the feature maps.
        downsample_channels (list[int]): Output channels
            of downsamplesample feature maps.
        upsample_strides (list[float]): Strides used to
            upsample the feature maps.
        upsample_channels (list[int]): Output channels
            of upsample feature maps.
        in_channels (int): Input channels of
            feature map.
        norm_cfg (dict): Configuration of norm layer.
    """

    def __init__(
            self,
            layer_nums,
            downsample_strides,
            downsample_channels,
            upsample_strides,
            upsample_channels,
            input_channels,
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    ):
        super(CenterPointRPN, self).__init__()
        self.downsample_strides = downsample_strides
        self.downsample_channels = downsample_channels
        self.layer_nums = layer_nums
        self.upsample_strides = upsample_strides
        self.upsample_channels = upsample_channels
        self.input_channels = input_channels
        self.norm_cfg = norm_cfg

        assert len(self.downsample_strides) == len(self.layer_nums)
        assert len(self.downsample_channels) == len(self.layer_nums)
        assert len(self.upsample_channels) == len(self.upsample_strides)

        self.upsample_start_idx = len(self.layer_nums) - len(
            self.upsample_strides)

        must_equal_list = []
        for i in range(len(self.upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(self.upsample_strides[i] / np.prod(
                self.downsample_strides[:i + self.upsample_start_idx + 1]))

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self.input_channels, *self.downsample_channels[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self.layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self.downsample_channels[i],
                layer_num,
                stride=self.downsample_strides[i],
            )
            blocks.append(block)
            if i - self.upsample_start_idx >= 0:
                stride = (self.upsample_strides[i - self.upsample_start_idx])
                if stride > 1:
                    deblock = nn.Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self.upsample_channels[i -
                                                   self.upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self.norm_cfg,
                            self.upsample_channels[i -
                                                   self.upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self.upsample_channels[i -
                                                   self.upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self.norm_cfg,
                            self.upsample_channels[i -
                                                   self.upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self.downsample_strides)
        if len(self.upsample_strides) > 0:
            factor /= self.upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block_list = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self.norm_cfg, planes)[1],
            nn.ReLU()
        ]

        for j in range(num_blocks):
            block_list.append(
                nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block_list.append(
                build_norm_layer(self.norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block_list.append(nn.ReLU())

        block = nn.Sequential(*block_list)
        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        """Forward of CenterPointRPN.

        Args:
            x (torch.Tensor): Input feature with the shape of [B, C, H, M].

        Returns:
            torch.Tensor: Concatenate features.
        """
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self.upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self.upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x
