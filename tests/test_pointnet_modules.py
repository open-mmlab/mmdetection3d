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
    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).cuda()
    # (B, C, N)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()

    # test forward
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 48, 16])
    assert inds.shape == torch.Size([1, 16])


def test_pointnet_sa_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import PointSAModule

    self = PointSAModule(
        num_point=16,
        radius=0.2,
        num_sample=8,
        mlp_channels=[12, 32],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max').cuda()

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
