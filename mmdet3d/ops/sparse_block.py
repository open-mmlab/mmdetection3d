from torch import nn

import mmdet3d.ops.spconv as spconv
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.ops import build_norm_layer
from mmdet.ops.conv import conv_cfg

conv_cfg.update({'SubMConv3d': spconv.SubMConv3d})


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 submanifold sparse convolution with padding.

    Args:
        in_planes (int): the number of input channels
        out_planes (int): the number of output channels
        stride (int): the stride of convolution
        indice_key (str): the indice key used for sparse tensor

    Returns:
        spconv.conv.SubMConv3d: 3x3 submanifold sparse convolution ops
    """
    # TODO: deprecate this class
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 submanifold sparse convolution with padding.

    Args:
        in_planes (int): the number of input channels
        out_planes (int): the number of output channels
        stride (int): the stride of convolution
        indice_key (str): the indice key used for sparse tensor

    Returns:
        spconv.conv.SubMConv3d: 1x1 submanifold sparse convolution ops
    """
    # TODO: deprecate this class
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


class SparseBasicBlockV0(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 indice_key=None,
                 norm_cfg=None):
        """Sparse basic block for PartA^2.

        Sparse basic block implemented with submanifold sparse convolution.
        """
        # TODO: deprecate this class
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key)
        norm_name1, norm_layer1 = build_norm_layer(norm_cfg, planes)
        self.bn1 = norm_layer1
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key)
        norm_name2, norm_layer2 = build_norm_layer(norm_cfg, planes)
        self.bn2 = norm_layer2
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class SparseBottleneckV0(spconv.SparseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 indice_key=None,
                 norm_fn=None):
        """Sparse bottleneck block for PartA^2.

        Bottleneck block implemented with submanifold sparse convolution.
        """
        # TODO: deprecate this class
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.conv2 = conv3x3(planes, planes, stride, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.conv3 = conv1x1(
            planes, planes * self.expansion, indice_key=indice_key)
        self.bn3 = norm_fn(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.relu(out.features)

        out = self.conv3(out)
        out.features = self.bn3(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class SparseBottleneck(Bottleneck, spconv.SparseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None):
        """Sparse bottleneck block for PartA^2.

        Bottleneck block implemented with submanifold sparse convolution.
        """
        spconv.SparseModule.__init__(self)
        Bottleneck.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.relu(out.features)

        out = self.conv3(out)
        out.features = self.bn3(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class SparseBasicBlock(BasicBlock, spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None):
        """Sparse basic block for PartA^2.

        Sparse basic block implemented with submanifold sparse convolution.
        """
        spconv.SparseModule.__init__(self)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'

        out = self.conv1(x)
        out.features = self.norm1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.norm2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out
