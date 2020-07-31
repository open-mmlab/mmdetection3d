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
        norm_cfg,
    ):
        super(CenterPointRPN, self).__init__()
        self._layer_strides = downsample_strides
        self._num_filters = downsample_channels
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = upsample_channels
        self._num_input_features = input_channels

        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(
            self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(self._upsample_strides[i] / np.prod(
                self._layer_strides[:i + self._upsample_start_idx + 1]))

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = nn.Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[
                                i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[
                                i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[
                                i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[
                                i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block_list = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            nn.ReLU()
        ]

        for j in range(num_blocks):
            block_list.append(
                nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block_list.append(
                build_norm_layer(self._norm_cfg, planes)[1],
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
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x
