import numpy as np
from mmcv.cnn import build_norm_layer
from torch import nn

from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.ops import spconv
from mmdet3d.ops.spconv.conv import SparseConv3d, SubMConv3d


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        kernal_size=3,
        stride=1,
        bias=False,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        self.conv1 = spconv.SubMConv3d(
            inplanes,
            planes,
            kernel_size=kernal_size,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


@MIDDLE_ENCODERS.register_module
class SpMiddleResNetFHD(nn.Module):
    """Sparse Middle ResNet FHD.

    Middle encoder used by CenterPoint.

    Args:
        num_input_features (int): Number of input features.
            Default: 128.

        norm_cfg (dict): Configuration of normalization.
            Default: dict(type='BN1d', eps=1e-3, momentum=0.01).
    """

    def __init__(self,
                 num_input_features=128,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 **kwargs):
        super(SpMiddleResNetFHD, self).__init__()

        self.dcn = None
        self.zero_init_residual = False

        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(
                num_input_features, 16, 3, bias=False, indice_key='res0'),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseBasicBlock(16, 16, indice_key='res0'),
            SparseBasicBlock(16, 16, indice_key='res0'),
            SparseConv3d(16, 32, 3, 2, padding=1,
                         bias=False),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, indice_key='res1'),
            SparseBasicBlock(32, 32, indice_key='res1'),
            SparseConv3d(32, 64, 3, 2, padding=1,
                         bias=False),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key='res2'),
            SparseBasicBlock(64, 64, indice_key='res2'),
            SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1],
                         bias=False),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, indice_key='res3'),
            SparseBasicBlock(128, 128, indice_key='res3'),
            SparseConv3d(128, 128, (3, 1, 1), (2, 1, 1),
                         bias=False),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        """Forward of SpMiddleResNetFHD.

        Args:
            voxel_features (torch.Tensor): Voxel features with the
                shape of [N, 5].
            coors (torch.Tensor): Voxel features with the shape of [N, 4].
            batch_size (int): Batch size.
            input_shape (np.ndarray): Shape of input.

        Returns:
            torch.Tensor: Result tensor.
        """
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret
