# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


def test_pointnet_sa_module_msg():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import PointSAModuleMSG

    self = PointSAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max').cuda()

    assert self.mlps[0].layer0.conv.in_channels == 12
    assert self.mlps[0].layer0.conv.out_channels == 16
    assert self.mlps[1].layer0.conv.in_channels == 12
    assert self.mlps[1].layer0.conv.out_channels == 32

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    # (B, N, 3)
    xyz = torch.from_numpy(xyz).view(1, -1, 3).cuda()
    # (B, C, N)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 48, 16])
    assert inds.shape == torch.Size([1, 16])

    # test D-FPS mod
    self = PointSAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        fps_mod=['D-FPS'],
        fps_sample_range_list=[-1]).cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 48, 16])
    assert inds.shape == torch.Size([1, 16])

    # test F-FPS mod
    self = PointSAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        fps_mod=['F-FPS'],
        fps_sample_range_list=[-1]).cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 48, 16])
    assert inds.shape == torch.Size([1, 16])

    # test FS mod
    self = PointSAModuleMSG(
        num_point=8,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        fps_mod=['FS'],
        fps_sample_range_list=[-1]).cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 48, 16])
    assert inds.shape == torch.Size([1, 16])

    # test using F-FPS mod and D-FPS mod simultaneously
    self = PointSAModuleMSG(
        num_point=[8, 12],
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        fps_mod=['F-FPS', 'D-FPS'],
        fps_sample_range_list=[64, -1]).cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 20, 3])
    assert new_features.shape == torch.Size([1, 48, 20])
    assert inds.shape == torch.Size([1, 20])

    # length of 'fps_mod' should be same as 'fps_sample_range_list'
    with pytest.raises(AssertionError):
        PointSAModuleMSG(
            num_point=8,
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            norm_cfg=dict(type='BN2d'),
            use_xyz=False,
            pool_mod='max',
            fps_mod=['F-FPS', 'D-FPS'],
            fps_sample_range_list=[-1]).cuda()

    # length of 'num_point' should be same as 'fps_sample_range_list'
    with pytest.raises(AssertionError):
        PointSAModuleMSG(
            num_point=[8, 8],
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            norm_cfg=dict(type='BN2d'),
            use_xyz=False,
            pool_mod='max',
            fps_mod=['F-FPS'],
            fps_sample_range_list=[-1]).cuda()


def test_pointnet_sa_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import build_sa_module
    sa_cfg = dict(
        type='PointSAModule',
        num_point=16,
        radius=0.2,
        num_sample=8,
        mlp_channels=[12, 32],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max')
    self = build_sa_module(sa_cfg).cuda()

    assert self.mlps[0].layer0.conv.in_channels == 15
    assert self.mlps[0].layer0.conv.out_channels == 32

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    # (B, N, 3)
    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).cuda()
    # (B, C, N)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 32, 16])
    assert inds.shape == torch.Size([1, 16])

    # can't set normalize_xyz when radius is None
    with pytest.raises(AssertionError):
        sa_cfg = dict(
            type='PointSAModule',
            num_point=16,
            radius=None,
            num_sample=8,
            mlp_channels=[12, 32],
            norm_cfg=dict(type='BN2d'),
            use_xyz=True,
            pool_mod='max',
            normalize_xyz=True)
        self = build_sa_module(sa_cfg)

    # test kNN sampling when radius is None
    sa_cfg['normalize_xyz'] = False
    self = build_sa_module(sa_cfg).cuda()

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).cuda()
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 32, 16])
    assert inds.shape == torch.Size([1, 16])


def test_pointnet_fp_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import PointFPModule

    self = PointFPModule(mlp_channels=[24, 16]).cuda()
    assert self.mlps.layer0.conv.in_channels == 24
    assert self.mlps.layer0.conv.out_channels == 16

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin',
                      np.float32).reshape((-1, 6))

    # (B, N, 3)
    xyz1 = torch.from_numpy(xyz[0::2, :3]).view(1, -1, 3).cuda()
    # (B, C1, N)
    features1 = xyz1.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()

    # (B, M, 3)
    xyz2 = torch.from_numpy(xyz[1::3, :3]).view(1, -1, 3).cuda()
    # (B, C2, N)
    features2 = xyz2.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()

    fp_features = self(xyz1, xyz2, features1, features2)
    assert fp_features.shape == torch.Size([1, 16, 50])
