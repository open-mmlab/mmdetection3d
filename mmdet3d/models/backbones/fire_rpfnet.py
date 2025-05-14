import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS


class FireBlock(nn.Module):
    """SqueezeNet‐style fire module with residual shortcut.

    Args:
        in_ch (int): #input channels.
        out_ch (int): #output channels of the expand concat.
        norm_cfg (dict): Normalisation config.
    """

    def __init__(self, in_ch, out_ch, norm_cfg):
        super().__init__()
        squeeze_ch = max(16, in_ch // 4)
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1, bias=False)
        self.squeeze_bn = build_norm_layer(norm_cfg, squeeze_ch)[1]
        # expand paths
        self.expand1x1 = nn.Conv2d(squeeze_ch, out_ch // 2, 1, bias=False)
        self.expand3x3 = nn.Conv2d(squeeze_ch, out_ch // 2, 3, padding=1, bias=False)
        self.expand_bn = build_norm_layer(norm_cfg, out_ch)[1]
        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                build_norm_layer(norm_cfg, out_ch)[1],
            )

    def forward(self, x):
        identity = x
        x = self.act(self.squeeze_bn(self.squeeze(x)))
        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)
        out = torch.cat([out1, out3], dim=1)
        out = self.expand_bn(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(out + identity)


class CBAM(nn.Module):
    """Channel & spatial attention."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        att_c = self.mlp(x).view(b, c, 1, 1)
        x = x * att_c
        att_s = self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        return x * att_s


@MODELS.register_module()
class FireRPFNet(nn.Module):
    """Residual FireNet backbone (SqueezeNet-inspired) with CBAM.

    Designed as a drop-in replacement for RPFNet in BEV pipelines.
    """

    def __init__(self,
                 in_channels=256,
                 layer_channels=(128, 256, 256, 256),
                 with_cbam=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super().__init__()
        layers = []
        ch = in_channels
        for out_ch in layer_channels:
            block = FireBlock(ch, out_ch, norm_cfg)
            stage = [block]
            if with_cbam:
                stage.append(CBAM(out_ch))
            layers.append(nn.Sequential(*stage))
            ch = out_ch
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return (x, )