"""FireRPFNet: Fire Module + CBAM Attention Backbones.

This module provides efficient backbones combining Fire modules (SqueezeNet-inspired)
with CBAM attention for both 2D image and 3D point cloud (BEV) feature extraction.

Variants:
    - FireRPFNet: Original BEV backbone
    - FireRPFNetV2: Enhanced BEV backbone with multi-scale support
    - FireRPFNet2D: 2D image backbone for multimodal detection

References:
    - SqueezeNet Fire Module: https://arxiv.org/abs/1602.07360
    - CBAM: https://arxiv.org/abs/1807.06521
    - MVXNet: https://arxiv.org/abs/1904.01649
"""

import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS


class FireBlock(nn.Module):
    """SqueezeNet-style fire module with residual shortcut for BEV features.

    Original implementation for point cloud BEV backbones (FireRPFNet, FireRPFNetV2).
    Squeezes from input channels for compatibility with trained models.

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels of the expand concat.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self, in_ch, out_ch, norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super().__init__()
        # Squeeze from INPUT channels (original BEV behavior)
        squeeze_ch = max(16, in_ch // 4)

        # Squeeze path (no stride, BEV maintains resolution)
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1, bias=False)
        self.squeeze_bn = build_norm_layer(norm_cfg, squeeze_ch)[1]

        # Expand paths (1x1 and 3x3 in parallel)
        self.expand1x1 = nn.Conv2d(squeeze_ch, out_ch // 2, 1, bias=False)
        self.expand3x3 = nn.Conv2d(squeeze_ch, out_ch // 2, 3, padding=1, bias=False)
        self.expand_bn = build_norm_layer(norm_cfg, out_ch)[1]

        self.act = nn.ReLU(inplace=True)

        # Residual connection with projection if needed
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                build_norm_layer(norm_cfg, out_ch)[1],
            )

    def forward(self, x):
        identity = x

        # Squeeze
        x = self.act(self.squeeze_bn(self.squeeze(x)))

        # Expand (parallel 1x1 and 3x3)
        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)
        out = torch.cat([out1, out3], dim=1)
        out = self.expand_bn(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.act(out + identity)


class FireBlock2D(nn.Module):
    """SqueezeNet-style fire module with residual shortcut for 2D images.

    Adapted for hierarchical image feature extraction with downsampling support.
    Squeezes from output channels for consistent behavior during resolution changes.

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels of the expand concat.
        stride (int): Stride for downsampling. Default: 1.
        norm_cfg (dict): Normalization config.
    """

    def __init__(self, in_ch, out_ch, stride=1, norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.stride = stride
        # Squeeze from OUTPUT channels (for stable behavior across resolution changes)
        squeeze_ch = max(16, out_ch // 4)

        # Squeeze path (with optional stride for downsampling)
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1, stride=stride, bias=False)
        self.squeeze_bn = build_norm_layer(norm_cfg, squeeze_ch)[1]

        # Expand paths (1x1 and 3x3 in parallel)
        self.expand1x1 = nn.Conv2d(squeeze_ch, out_ch // 2, 1, bias=False)
        self.expand3x3 = nn.Conv2d(squeeze_ch, out_ch // 2, 3, padding=1, bias=False)
        self.expand_bn = build_norm_layer(norm_cfg, out_ch)[1]

        self.act = nn.ReLU(inplace=True)

        # Residual connection with projection if needed
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, out_ch)[1],
            )

    def forward(self, x):
        identity = x

        # Squeeze
        x = self.act(self.squeeze_bn(self.squeeze(x)))

        # Expand (parallel 1x1 and 3x3)
        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)
        out = torch.cat([out1, out3], dim=1)
        out = self.expand_bn(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.act(out + identity)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Applies sequential channel and spatial attention to input features.

    Args:
        ch (int): Number of input channels.
        reduction (int): Channel reduction ratio for MLP. Default: 16.

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """

    def __init__(self, ch, reduction=16):
        super().__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention
        att_c = self.channel_att(x).view(b, c, 1, 1)
        x = x * att_c

        # Spatial attention
        att_s = self.spatial_att(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        )
        return x * att_s


# =============================================================================
# BEV Backbones (for Point Cloud)
# =============================================================================

@MODELS.register_module()
class FireRPFNet(nn.Module):
    """Residual FireNet backbone (SqueezeNet-inspired) with CBAM.

    first version designed as a drop-in replacement for RPFNet in BEV pipelines.
    Processes BEV features from sparse 3D convolution without downsampling.

    Args:
        in_channels (int): Input channels. Default: 256.
        out_channels (tuple[int]): Output channels for each stage.
            Default: (128, 256, 256, 256).
        with_cbam (bool): Whether to use CBAM attention. Default: True.
        norm_cfg (dict): Normalization config.
            Default: dict(type='BN', eps=1e-3, momentum=0.01).
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=(128, 256, 256, 256),
                 with_cbam=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super().__init__()
        layers = []
        ch = in_channels
        for out_ch in out_channels:
            block = FireBlock(ch, out_ch, norm_cfg=norm_cfg)
            stage = [block]
            if with_cbam:
                stage.append(CBAM(out_ch))
            layers.append(nn.Sequential(*stage))
            ch = out_ch
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): BEV features (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Single-element tuple with last stage output.
        """
        for stage in self.stages:
            x = stage(x)
        return (x, )


@MODELS.register_module()
class FireRPFNetV2(nn.Module):
    """Enhanced Residual FireNet backbone with multi-scale support.

    Designed as a drop-in replacement for RPFNet/SECOND in BEV pipelines.
    Can output single-scale or multi-scale features for use with/without FPN necks.

    Args:
        in_channels (int): Input channels. Default: 256.
        out_channels (tuple[int] | list[int]): Output channels for each stage.
            Default: (128, 256, 256, 256).
        with_cbam (bool): Whether to use CBAM attention after each stage.
            Default: True.
        multi_scale_output (bool): If True, returns multi-scale features from all stages
            (for use with SECONDFPN neck). If False, returns only the last stage output
            (backward compatible, for use without neck). Default: False.
        norm_cfg (dict): Normalization config.
            Default: dict(type='BN', eps=1e-3, momentum=0.01).

    Example:
        >>> # Single-scale output (no neck)
        >>> pts_backbone = dict(
        ...     type='FireRPFNetV2',
        ...     in_channels=256,
        ...     out_channels=[128, 256, 256, 256],
        ...     multi_scale_output=False)

        >>> # Multi-scale output (with SECONDFPN)
        >>> pts_backbone = dict(
        ...     type='FireRPFNetV2',
        ...     in_channels=256,
        ...     out_channels=[128, 256, 256, 256],
        ...     multi_scale_output=True)
        >>> pts_neck = dict(
        ...     type='SECONDFPN',
        ...     in_channels=[128, 256, 256, 256],
        ...     upsample_strides=[1, 2, 4, 8],
        ...     out_channels=[128, 128, 128, 128])
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=(128, 256, 256, 256),
                 with_cbam=True,
                 multi_scale_output=False,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.multi_scale_output = multi_scale_output
        layers = []
        ch = in_channels
        for out_ch in out_channels:
            block = FireBlock(ch, out_ch, norm_cfg=norm_cfg)
            stage = [block]
            if with_cbam:
                stage.append(CBAM(out_ch))
            layers.append(nn.Sequential(*stage))
            ch = out_ch
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): BEV features (N, C, H, W).

        Returns:
            tuple[torch.Tensor]:
                - If multi_scale_output=False: Single-element tuple with last stage output
                - If multi_scale_output=True: Multi-element tuple with all stage outputs
        """
        if self.multi_scale_output:
            # Return multi-scale features for FPN neck
            outs = []
            for stage in self.stages:
                x = stage(x)
                outs.append(x)
            return tuple(outs)
        else:
            # Return only last stage (backward compatible)
            for stage in self.stages:
                x = stage(x)
            return (x, )


# =============================================================================
# 2D Image Backbone
# =============================================================================

@MODELS.register_module()
class FireRPFNet2D(nn.Module):
    """FireRPFNet2D: Efficient 2D image backbone for multimodal 3D detection.

    This backbone uses Fire modules (SqueezeNet-inspired) with CBAM attention
    for efficient feature extraction from RGB images. It outputs multi-scale features
    suitable for FPN necks in MVXNet-style architectures.

    Architecture:
        - Stem: Conv 7×7 stride=2 + MaxPool → H/4, W/4
        - Stage 1: Fire blocks (stride=1) → H/4, W/4
        - Stage 2: Fire blocks (stride=2 in first) → H/8, W/8
        - Stage 3: Fire blocks (stride=2 in first) → H/16, W/16
        - Stage 4: Fire blocks (stride=2 in first) → H/32, W/32
        - Each block optionally followed by CBAM attention

    Args:
        in_channels (int): Input image channels (typically 3 for RGB). Default: 3.
        out_channels (tuple[int]): Output channels for each stage.
            Default: (64, 128, 256, 512).
        blocks_per_stage (tuple[int]): Number of Fire blocks per stage.
            Default: (2, 2, 2, 2).
        with_cbam (bool): Whether to use CBAM attention after each block.
            Default: True.
        stem_channels (int): Channels in stem conv. Default: 64.
        out_indices (tuple[int]): Output feature indices for multi-scale.
            Default: (0, 1, 2, 3) - all stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any stages. Default: -1.
        norm_cfg (dict): Normalization config.
            Default: dict(type='BN', eps=1e-3, momentum=0.01).
        norm_eval (bool): Whether to set norm layers to eval mode. Default: False.

    Example:
        >>> # Standard configuration
        >>> img_backbone = dict(
        ...     type='FireRPFNet2D',
        ...     in_channels=3,
        ...     out_channels=[64, 128, 256, 512],
        ...     stem_channels=64,
        ...     with_cbam=True)

        >>> # Lightweight configuration (~40% fewer params)
        >>> img_backbone = dict(
        ...     type='FireRPFNet2D',
        ...     out_channels=[48, 96, 192, 384],
        ...     stem_channels=48,
        ...     with_cbam=True)

        >>> # Without attention
        >>> img_backbone = dict(
        ...     type='FireRPFNet2D',
        ...     out_channels=[64, 128, 256, 512],
        ...     with_cbam=False)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2),
                 with_cbam=True,
                 stem_channels=64,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 norm_eval=False):
        super().__init__()

        assert len(out_channels) == len(blocks_per_stage), \
            "out_channels and blocks_per_stage must have same length"

        self.num_stages = len(out_channels)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cbam = with_cbam

        # Stem: initial convolution to lift channels (H/4, W/4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2,
                     padding=3, bias=False),
            build_norm_layer(norm_cfg, stem_channels)[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = stem_channels

        for stage_idx, (out_ch, num_blocks) in enumerate(zip(out_channels, blocks_per_stage)):
            blocks = []
            for block_idx in range(num_blocks):
                # First block of stages 1-3 uses stride=2 for downsampling
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1

                # Fire block (use FireBlock2D for image backbone)
                fire_block = FireBlock2D(in_ch, out_ch, stride=stride, norm_cfg=norm_cfg)
                blocks.append(fire_block)

                # Optional CBAM attention
                if with_cbam:
                    blocks.append(CBAM(out_ch))

                in_ch = out_ch

            self.stages.append(nn.Sequential(*blocks))

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages parameters and set to eval mode."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            if i < len(self.stages):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input images (N, C, H, W), typically (N, 3, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale feature maps from selected stages.
                Each tensor has shape (N, C_i, H_i, W_i).
        """
        x = self.stem(x)  # Initial downsampling: H/4, W/4

        outs = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        """Set the module in training mode."""
        super(FireRPFNet2D, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
