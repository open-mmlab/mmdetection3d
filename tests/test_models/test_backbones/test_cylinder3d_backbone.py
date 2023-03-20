# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.registry import MODELS


def test_cylinder3d():
    if not torch.cuda.is_available():
        pytest.skip()
    cfg = dict(
        type='Asymm3DSpconv',
        grid_size=[48, 32, 4],
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1))
    self = MODELS.build(cfg)
    self.cuda()

    batch_size = 1
    coorx = torch.randint(0, 48, (50, 1))
    coory = torch.randint(0, 36, (50, 1))
    coorz = torch.randint(0, 4, (50, 1))
    coorbatch = torch.zeros(50, 1)
    coors = torch.cat([coorbatch, coorx, coory, coorz], dim=1).cuda()
    voxel_features = torch.rand(50, 16).cuda()

    # test forward
    feature = self(voxel_features, coors, batch_size)

    assert feature.features.shape == (50, 128)
    assert feature.indices.data.shape == (50, 4)
