# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmengine.device import is_musa_available, is_cuda_available


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not is_cuda_available(), reason="requires CUDA support"
            ),
        ),
        pytest.param(
            "musa",
            marks=pytest.mark.skipif(
                not is_musa_available(), reason="requires MUSA support"
            ),
        ),
    ],
)
def test_pointnet_sa_module_msg(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()
    from mmdet3d.models.layers import PointSAModuleMSG

    self = PointSAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max').to(device)

    assert self.mlps[0].layer0.conv.in_channels == 12
    assert self.mlps[0].layer0.conv.out_channels == 16
    assert self.mlps[1].layer0.conv.in_channels == 12
    assert self.mlps[1].layer0.conv.out_channels == 32

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    # (B, N, 3)
    xyz = torch.from_numpy(xyz).view(1, -1, 3).to(device)
    # (B, C, N)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().to(device)

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
        fps_sample_range_list=[-1]).to(device)

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
        fps_sample_range_list=[-1]).to(device)

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
        fps_sample_range_list=[-1]).to(device)

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
        fps_sample_range_list=[64, -1]).to(device)

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 20, 3])
    assert new_features.shape == torch.Size([1, 48, 20])
    assert inds.shape == torch.Size([1, 20])

    # test num_points = None
    self = PointSAModuleMSG(
        num_point=None,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max').to(device)

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_features.shape == torch.Size([1, 48, 1])

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
            fps_sample_range_list=[-1]).to(device)

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
            fps_sample_range_list=[-1]).to(device)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not is_cuda_available(), reason="requires CUDA support"
            ),
        ),
        pytest.param(
            "musa",
            marks=pytest.mark.skipif(
                not is_musa_available(), reason="requires MUSA support"
            ),
        ),
    ],
)
def test_pointnet_sa_module(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()
    from mmdet3d.models.layers import build_sa_module
    sa_cfg = dict(
        type='PointSAModule',
        num_point=16,
        radius=0.2,
        num_sample=8,
        mlp_channels=[12, 32],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max')
    self = build_sa_module(sa_cfg).to(device)

    assert self.mlps[0].layer0.conv.in_channels == 15
    assert self.mlps[0].layer0.conv.out_channels == 32

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    # (B, N, 3)
    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).to(device)
    # (B, C, N)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().to(device)

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
    self = build_sa_module(sa_cfg).to(device)

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).to(device)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().to(device)
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 32, 16])
    assert inds.shape == torch.Size([1, 16])
