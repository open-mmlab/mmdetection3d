# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.models.layers import SparseBasicBlock
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConv3d, SparseInverseConv3d, SubMConv3d
else:
    from mmcv.ops import SparseConv3d, SparseInverseConv3d, SubMConv3d


def test_SparseUNet():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    from mmdet3d.models.middle_encoders.sparse_unet import SparseUNet
    self = SparseUNet(in_channels=4, sparse_shape=[41, 1600, 1408]).cuda()

    # test encoder layers
    assert len(self.encoder_layers) == 4
    assert self.encoder_layers.encoder_layer1[0][0].in_channels == 16
    assert self.encoder_layers.encoder_layer1[0][0].out_channels == 16
    assert isinstance(self.encoder_layers.encoder_layer1[0][0], SubMConv3d)
    assert isinstance(self.encoder_layers.encoder_layer1[0][1],
                      torch.nn.modules.batchnorm.BatchNorm1d)
    assert isinstance(self.encoder_layers.encoder_layer1[0][2],
                      torch.nn.modules.activation.ReLU)
    assert self.encoder_layers.encoder_layer4[0][0].in_channels == 64
    assert self.encoder_layers.encoder_layer4[0][0].out_channels == 64
    assert isinstance(self.encoder_layers.encoder_layer4[0][0], SparseConv3d)
    assert isinstance(self.encoder_layers.encoder_layer4[2][0], SubMConv3d)

    # test decoder layers
    assert isinstance(self.lateral_layer1, SparseBasicBlock)
    assert isinstance(self.merge_layer1[0], SubMConv3d)
    assert isinstance(self.upsample_layer1[0], SubMConv3d)
    assert isinstance(self.upsample_layer2[0], SparseInverseConv3d)

    voxel_features = torch.tensor(
        [[6.56126, 0.9648336, -1.7339306, 0.315],
         [6.8162713, -2.480431, -1.3616394, 0.36],
         [11.643568, -4.744306, -1.3580885, 0.16],
         [23.482342, 6.5036807, 0.5806964, 0.35]],
        dtype=torch.float32).cuda()  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32).cuda()  # n, 4(batch, ind_x, ind_y, ind_z)

    unet_ret_dict = self.forward(voxel_features, coordinates, 2)
    seg_features = unet_ret_dict['seg_features']
    spatial_features = unet_ret_dict['spatial_features']

    assert seg_features.shape == torch.Size([4, 16])
    assert spatial_features.shape == torch.Size([2, 256, 200, 176])
