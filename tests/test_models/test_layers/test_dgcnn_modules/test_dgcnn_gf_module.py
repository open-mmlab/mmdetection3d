# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


def test_dgcnn_gf_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.models.layers import DGCNNGFModule

    self = DGCNNGFModule(
        mlp_channels=[18, 64, 64],
        num_sample=20,
        knn_mode='D-KNN',
        radius=None,
        norm_cfg=dict(type='BN2d'),
        act_cfg=dict(type='ReLU'),
        pool_mode='max').cuda()

    assert self.mlps[0].layer0.conv.in_channels == 18
    assert self.mlps[0].layer0.conv.out_channels == 64

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    # (B, N, C)
    xyz = torch.from_numpy(xyz).view(1, -1, 3).cuda()
    points = xyz.repeat([1, 1, 3])

    # test forward
    new_points = self(points)

    assert new_points.shape == torch.Size([1, 200, 64])

    # test F-KNN mod
    self = DGCNNGFModule(
        mlp_channels=[6, 64, 64],
        num_sample=20,
        knn_mode='F-KNN',
        radius=None,
        norm_cfg=dict(type='BN2d'),
        act_cfg=dict(type='ReLU'),
        pool_mode='max').cuda()

    # test forward
    new_points = self(xyz)
    assert new_points.shape == torch.Size([1, 200, 64])

    # test ball query
    self = DGCNNGFModule(
        mlp_channels=[6, 64, 64],
        num_sample=20,
        knn_mode='F-KNN',
        radius=0.2,
        norm_cfg=dict(type='BN2d'),
        act_cfg=dict(type='ReLU'),
        pool_mode='max').cuda()
