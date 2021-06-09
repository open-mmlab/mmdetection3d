from torch import nn

from mmdet.models import NECKS


@NECKS.register_module()
class OutdoorImVoxelNeck(nn.Module):
    """Neck for ImVoxelNet outdoor scenario.

    Args:
        in_channels (int): Input channels of multi-scale feature map.
        out_channels (int): Output channels of multi-scale feature map.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels),
            ConvBlock(in_channels, in_channels * 2, stride=(1, 1, 2)),
            ResBlock(in_channels * 2),
            ConvBlock(in_channels * 2, in_channels * 4, stride=(1, 1, 2)),
            ResBlock(in_channels * 4),
            ConvBlock(in_channels * 4, out_channels, padding=(1, 1, 0)))

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


class ConvBlock(nn.Module):
    """3d convolution block for ImVoxelNeck.

    Args:
        in_channels (int): Input channels of a feature map.
        out_channels (int): Output channels of a feature map.
        stride (int): Stride of 3d convolution.
        padding (int): Padding of 3d convolution.
        activation (bool): Whether to use ReLU.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 activation=True):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, 3, stride=stride, padding=padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """3d residual block for ImVoxelNeck.

    Args:
        n_channels (int): Input channels of a feature map.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv0 = ConvBlock(n_channels, n_channels)
        self.conv1 = ConvBlock(n_channels, n_channels, activation=False)
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
        x = identity + x
        x = self.activation(x)
        return x
