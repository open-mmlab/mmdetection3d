import math
import numpy as np
from mmcv.cnn import build_norm_layer
from mmcv.ops import ModulatedDeformConv2dPack
from torch import nn as nn

from mmdet.models.builder import NECKS


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    """DCNv2 specially designed for DLA Neck.

    Args:
        chi (int): Number of input channels.
        cho (int): Number of output channels.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self, chi, cho, norm_cfg=None):
        super(DeformConv, self).__init__()
        self.norm = build_norm_layer(norm_cfg, cho)[1]
        self.relu = nn.ReLU(inplace=True)
        self.deform_conv = ModulatedDeformConv2dPack(
            chi,
            cho,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=1)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class IDAUp(nn.Module):
    """IDAUp module for upsamling different scale's features to same scale.

    Args:
        o (int): Number of output channels for DeformConv.
        channels (List[int]): List of input channels of different scales.
        up_f (List[int]): List of size of the convolving kernel of
                          different scales.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self, o, channels, up_f, norm_cfg=None):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o, norm_cfg)
            node = DeformConv(o, o, norm_cfg)

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):

    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 norm_cfg=nn.BatchNorm2d):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j],
                      norm_cfg))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@NECKS.register_module()
class DLA_Neck(nn.Module):
    """DLA Neck.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        start_level (int): the scale level where upsampling starts. Default: 2.
        end_level (int): the scale level where upsampling ends. Default: 5.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5,
                 norm_cfg=None):
        super(DLA_Neck, self).__init__()
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUp(
            startp=self.start_level,  # 2
            channels=in_channels[self.start_level:],
            scales=scales,  # [1, 2, 4, 8]
            norm_cfg=norm_cfg)
        self.ida_up = IDAUp(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)], norm_cfg)

    def forward(self, x):
        x = self.dla_up(x)
        y = []
        for i in range(self.end_level - self.start_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return y[-1]

    def init_weights(self, pretrained=True):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ModulatedDeformConv2dPack):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                        1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
