import torch

import mmdet3d.ops.spconv as spconv
from mmdet3d.ops import SparseBasicBlock, SparseBasicBlockV0


def test_SparseUnet():
    from mmdet3d.models.middle_encoders.sparse_unet import SparseUnet
    self = SparseUnet(
        in_channels=4, output_shape=[41, 1600, 1408], pre_act=False)

    # test encoder layers
    assert len(self.encoder_layers) == 4
    assert self.encoder_layers.encoder_layer1[0][0].in_channels == 16
    assert self.encoder_layers.encoder_layer1[0][0].out_channels == 16
    assert isinstance(self.encoder_layers.encoder_layer1[0][0],
                      spconv.conv.SubMConv3d)
    assert isinstance(self.encoder_layers.encoder_layer1[0][1],
                      torch.nn.modules.batchnorm.BatchNorm1d)
    assert isinstance(self.encoder_layers.encoder_layer1[0][2],
                      torch.nn.modules.activation.ReLU)
    assert self.encoder_layers.encoder_layer4[0][0].in_channels == 64
    assert self.encoder_layers.encoder_layer4[0][0].out_channels == 64
    assert isinstance(self.encoder_layers.encoder_layer4[0][0],
                      spconv.conv.SparseConv3d)
    assert isinstance(self.encoder_layers.encoder_layer4[2][0],
                      spconv.conv.SubMConv3d)

    # test decoder layers
    assert isinstance(self.lateral_layer1, SparseBasicBlock)
    assert isinstance(self.merge_layer1[0], spconv.conv.SubMConv3d)
    assert isinstance(self.upsample_layer1[0], spconv.conv.SubMConv3d)
    assert isinstance(self.upsample_layer2[0], spconv.conv.SparseInverseConv3d)

    voxel_features = torch.tensor([[6.56126, 0.9648336, -1.7339306, 0.315],
                                   [6.8162713, -2.480431, -1.3616394, 0.36],
                                   [11.643568, -4.744306, -1.3580885, 0.16],
                                   [23.482342, 6.5036807, 0.5806964, 0.35]],
                                  dtype=torch.float32)  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32)  # n, 4(batch, ind_x, ind_y, ind_z)

    unet_ret_dict = self.forward(voxel_features, coordinates, 2)
    seg_features = unet_ret_dict['seg_features']
    spatial_features = unet_ret_dict['spatial_features']

    assert seg_features.shape == torch.Size([4, 16])
    assert spatial_features.shape == torch.Size([2, 256, 200, 176])


def test_SparseBasicBlock():
    voxel_features = torch.tensor([[6.56126, 0.9648336, -1.7339306, 0.315],
                                   [6.8162713, -2.480431, -1.3616394, 0.36],
                                   [11.643568, -4.744306, -1.3580885, 0.16],
                                   [23.482342, 6.5036807, 0.5806964, 0.35]],
                                  dtype=torch.float32)  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32)  # n, 4(batch, ind_x, ind_y, ind_z)

    # test v0
    self = SparseBasicBlockV0(
        4,
        4,
        indice_key='subm0',
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))
    input_sp_tensor = spconv.SparseConvTensor(voxel_features, coordinates,
                                              [41, 1600, 1408], 2)
    out_features = self(input_sp_tensor)
    assert out_features.features.shape == torch.Size([4, 4])

    # test
    input_sp_tensor = spconv.SparseConvTensor(voxel_features, coordinates,
                                              [41, 1600, 1408], 2)
    self = SparseBasicBlock(
        4,
        4,
        conv_cfg=dict(type='SubMConv3d', indice_key='subm1'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))
    # test conv and bn layer
    assert isinstance(self.conv1, spconv.conv.SubMConv3d)
    assert self.conv1.in_channels == 4
    assert self.conv1.out_channels == 4
    assert isinstance(self.conv2, spconv.conv.SubMConv3d)
    assert self.conv2.out_channels == 4
    assert self.conv2.out_channels == 4
    assert self.bn1.eps == 1e-3
    assert self.bn1.momentum == 0.01

    out_features = self(input_sp_tensor)
    assert out_features.features.shape == torch.Size([4, 4])
