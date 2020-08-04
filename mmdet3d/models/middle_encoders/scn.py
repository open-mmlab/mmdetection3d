import numpy as np
from mmcv.cnn import build_norm_layer
from torch import nn

from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.ops import SparseBasicBlock, spconv
from mmdet3d.ops.spconv.conv import SparseConv3d, SubMConv3d


@MIDDLE_ENCODERS.register_module
class SpMiddleResNetFHD(nn.Module):
    """Sparse Middle ResNet FHD.

    Middle encoder used by CenterPoint.

    Args:
        num_input_features (int): Number of input features.
            Default: 128.
        norm_cfg (dict): Config of normalization.
            Default: dict(type='BN1d', eps=1e-3, momentum=0.01).
        conv_cfg (dict): Config of conv.
            Default: dict(type='SubMConv3d').
    """

    def __init__(self,
                 num_input_features=128,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='SubMConv3d'),
                 **kwargs):
        super(SpMiddleResNetFHD, self).__init__()

        self.dcn = None
        self.zero_init_residual = False
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseConv3d(16, 32, 3, 2, padding=1,
                         bias=False),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseConv3d(32, 64, 3, 2, padding=1,
                         bias=False),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1],
                         bias=False),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
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
