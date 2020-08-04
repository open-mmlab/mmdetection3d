import numpy as np
import pytest
import torch

from mmdet3d.models.builder import build_middle_encoder


def test_sp_middle_resnet_FHD():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    sp_middle_resnet_FHD_cfg = dict(
        type='SpMiddleResNetFHD', num_input_features=5, ds_factor=8)

    sp_middle_resnet_FHD = build_middle_encoder(
        sp_middle_resnet_FHD_cfg).cuda()

    voxel_features = torch.rand([207842, 5]).cuda()
    coors = torch.randint(0, 4, [207842, 4]).cuda()

    ret = sp_middle_resnet_FHD(voxel_features, coors, 4,
                               np.array([1024, 1024, 40]))
    assert ret.shape == torch.Size([4, 256, 128, 128])


def test_sparse_encoder():
    middle_encoder = dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1,
                                                                       1)),
        option='basicblock')

    sparse_encoder = build_middle_encoder(middle_encoder)
    print(sparse_encoder)
