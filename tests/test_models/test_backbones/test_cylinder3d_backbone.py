# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS


def test_cylinder3d():
    if not torch.cuda.is_available():
        pytest.skip()
    cfg = dict(
        type='Asymm3DSpconv',
        grid_size=[480, 360, 32],
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1))
    self = MODELS.build(cfg)
    self.cuda()

    pts_feats = torch.rand(100, 16).cuda()
    coorx = torch.randint(0, 480, (100, 1)).int().cuda()
    coory = torch.randint(0, 360, (100, 1)).int().cuda()
    coorz = torch.randint(0, 32, (100, 1)).int().cuda()
    coorbatch = torch.zeros(100, 1).int().cuda()
    coors = torch.cat([coorbatch, coorx, coory, coorz], dim=1)
    voxel_feats, voxel_coors, point2voxel_map = dynamic_scatter_3d(
        pts_feats, coors)
    voxel_dict = dict(
        coors=coors, voxel_coors=voxel_coors, voxel_feats=voxel_feats)

    # test forward
    voxel_dict = self(voxel_dict)

    assert voxel_dict['voxel_feats'].features.shape == (voxel_coors.shape[0],
                                                        128)
    assert voxel_dict['voxel_feats'].indices.data.shape == (
        voxel_coors.shape[0], 4)
