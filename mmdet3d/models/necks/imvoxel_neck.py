# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class OutdoorImVoxelNeck(BaseModule):
    """Neck for ImVoxelNet outdoor scenario.

    Args:
        in_channels (int): Number of channels in an input tensor.
        out_channels (int): Number of channels in all output tensors.
    """

    def __init__(self, in_channels, out_channels):
        super(OutdoorImVoxelNeck, self).__init__()
        self.model = nn.Sequential(
            ResModule(in_channels, in_channels),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=3,
                stride=(1, 1, 2),
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels * 2, in_channels * 2),
            ConvModule(
                in_channels=in_channels * 2,
                out_channels=in_channels * 4,
                kernel_size=3,
                stride=(1, 1, 2),
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels * 4, in_channels * 4),
            ConvModule(
                in_channels=in_channels * 4,
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 1, 0),
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True)))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        x = self.model.forward(x)
        assert x.shape[-1] == 1
        # Anchor3DHead axis order is (y, x).
        return [x[..., 0].transpose(-1, -2)]

    def init_weights(self):
        """Initialize weights of neck."""
        pass


@MODELS.register_module()
class IndoorImVoxelNeck(BaseModule):
    """Neck for ImVoxelNet outdoor scenario.

    Args:
        in_channels (int): Number of channels in an input tensor.
        out_channels (int): Number of channels in all output tensors.
        n_blocks (list[int]): Number of blocks for each feature level.
    """

    def __init__(self, in_channels, out_channels, n_blocks):
        super(IndoorImVoxelNeck, self).__init__()
        self.n_scales = len(n_blocks)
        n_channels = in_channels
        for i in range(len(n_blocks)):
            stride = 1 if i == 0 else 2
            self.__setattr__(f'down_layer_{i}',
                             self._make_layer(stride, n_channels, n_blocks[i]))
            n_channels = n_channels * stride
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(n_channels, n_channels // 2))
            self.__setattr__(f'out_block_{i}',
                             self._make_block(n_channels, out_channels))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_xi, N_yi, N_zi).
        """
        down_outs = []
        for i in range(self.n_scales):
            x = self.__getattr__(f'down_layer_{i}')(x)
            down_outs.append(x)
        outs = []
        for i in range(self.n_scales - 1, -1, -1):
            if i < self.n_scales - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = down_outs[i] + x
            out = self.__getattr__(f'out_block_{i}')(x)
            outs.append(out)
        return outs[::-1]

    @staticmethod
    def _make_layer(stride, n_channels, n_blocks):
        """Make a layer from several residual blocks.

        Args:
            stride (int): Stride of the first residual block.
            n_channels (int): Number of channels of the first residual block.
            n_blocks (int): Number of residual blocks.

        Returns:
            torch.nn.Module: With several residual blocks.
        """
        blocks = []
        for i in range(n_blocks):
            if i == 0 and stride != 1:
                blocks.append(ResModule(n_channels, n_channels * 2, stride))
                n_channels = n_channels * 2
            else:
                blocks.append(ResModule(n_channels, n_channels))
        return nn.Sequential(*blocks)

    @staticmethod
    def _make_block(in_channels, out_channels):
        """Make a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: Convolutional block.
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        """Make upsampling convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: Upsampling convolutional block.
        """

        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))


class ResModule(nn.Module):
    """3d residual block for ImVoxelNeck.

    Args:
        in_channels (int): Number of channels in input tensor.
        out_channels (int): Number of channels in output tensor.
        stride (int, optional): Stride of the block. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=None)
        if stride != 1:
            self.downsample = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=None)
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        if self.stride != 1:
            identity = self.downsample(identity)
        x = x + identity
        x = self.activation(x)
        return x
