from torch import nn
from mmdet.models import NECKS


@NECKS.register_module()
class OutdoorImVoxelNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            # todo: padding should be (1, 1, 0) here
            self._get_conv(in_channels * 4, out_channels, 1, 0)
        )

    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model.forward(x)
        assert x.shape[-1] == 1
        return [x[..., 0].transpose(-1, -2)]

    def init_weights(self):
        pass


# Everything below is copied from https://github.com/magicleap/Atlas/blob/master/atlas/backbone3d.py
def get_norm_3d(norm, out_channels):
    """ Get a normalization module for 3D tensors
    Args:
        norm: (str or callable)
        out_channels
    Returns:
        nn.Module or None: the normalization layer
    """

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock3d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(BasicBlock3d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_3d(norm, planes)
        self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_3d(norm, planes)
        self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
