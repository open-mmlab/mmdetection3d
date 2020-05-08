from functools import partial

import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint

from mmdet.models import BACKBONES


class Empty(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


@BACKBONES.register_module()
class SECOND(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """

    def __init__(self,
                 in_channels=128,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)

        if norm_cfg is not None:
            Conv2d = partial(nn.Conv2d, bias=False)
        else:
            Conv2d = partial(nn.Conv2d, bias=True)

        in_filters = [in_channels, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []

        for i, layer_num in enumerate(layer_nums):
            norm_layer = (
                build_norm_layer(norm_cfg, num_filters[i])[1]
                if norm_cfg is not None else Empty)
            block = [
                nn.ZeroPad2d(1),
                Conv2d(
                    in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                norm_layer,
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                norm_layer = (
                    build_norm_layer(norm_cfg, num_filters[i])[1]
                    if norm_cfg is not None else Empty)
                block.append(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.append(norm_layer)
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
