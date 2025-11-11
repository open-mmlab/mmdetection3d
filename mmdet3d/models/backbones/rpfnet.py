import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS


class BasicBlock(nn.Module):
    """Simple residual 2-D conv block used in PillarNet-LTS (RPFN)."""

    def __init__(self, in_channels, out_channels, norm_cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.act(out + identity)
        return out


class CBAM(nn.Module):
    """Lightweight CBAM attention (channel + spatial)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # channel attention
        b, c, _, _ = x.size()
        channel_att = self.mlp(x).view(b, c, 1, 1)
        x = x * channel_att
        # spatial attention
        spatial_att = self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        x = x * spatial_att
        return x


@MODELS.register_module()
class RPFNet(nn.Module):
    """Residual Pillar Feature Network backbone (simplified).

    Args:
        in_channels (int): #Channels of input BEV feature map (from SparseEncoder).
        layer_channels (list[int]): Output channels for each residual stage.
        with_cbam (bool): If True, append a CBAM after each stage.
        norm_cfg (dict): Norm config dict.
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
            block = BasicBlock(ch, out_ch, norm_cfg)
            stage = [block]
            if with_cbam:
                stage.append(CBAM(out_ch))
            layers.append(nn.Sequential(*stage))
            ch = out_ch
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, C, H, W) BEV feature map
        for stage in self.stages:
            x = stage(x)
        # Anchor3DHead expects a tuple/list of multi-scale features.
        # We return a single-scale tuple to stay compatible.
        return (x, )
