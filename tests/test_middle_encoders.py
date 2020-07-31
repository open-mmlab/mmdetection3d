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
