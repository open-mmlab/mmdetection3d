# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


def test_paconv_sa_module_msg():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import PAConvSAModuleMSG

    # paconv_num_kernels should have same length as mlp_channels
    with pytest.raises(AssertionError):
        self = PAConvSAModuleMSG(
            num_point=16,
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            paconv_num_kernels=[[4]]).cuda()

    # paconv_num_kernels inner num should match as mlp_channels
    with pytest.raises(AssertionError):
        self = PAConvSAModuleMSG(
            num_point=16,
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            paconv_num_kernels=[[4, 4], [8, 8]]).cuda()

    self = PAConvSAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        paconv_num_kernels=[[4], [8]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        paconv_kernel_input='w_neighbor').cuda()

    assert self.mlps[0].layer0.in_channels == 12 * 2
    assert self.mlps[0].layer0.out_channels == 16
    assert self.mlps[1].layer0.in_channels == 12 * 2
    assert self.mlps[1].layer0.out_channels == 32
    assert self.mlps[0].layer0.bn.num_features == 16
    assert self.mlps[1].layer0.bn.num_features == 32

    assert self.mlps[0].layer0.scorenet.mlps.layer0.conv.in_channels == 7
    assert self.mlps[0].layer0.scorenet.mlps.layer3.conv.out_channels == 4
    assert self.mlps[1].layer0.scorenet.mlps.layer0.conv.in_channels == 7
    assert self.mlps[1].layer0.scorenet.mlps.layer3.conv.out_channels == 8

    # last conv in ScoreNet has neither bn nor relu
    with pytest.raises(AttributeError):
        _ = self.mlps[0].layer0.scorenet.mlps.layer3.bn
    with pytest.raises(AttributeError):
        _ = self.mlps[0].layer0.scorenet.mlps.layer3.activate

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

    # test with identity kernel input
    self = PAConvSAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        paconv_num_kernels=[[4], [8]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        paconv_kernel_input='identity').cuda()

    assert self.mlps[0].layer0.in_channels == 12 * 1
    assert self.mlps[0].layer0.out_channels == 16
    assert self.mlps[0].layer0.num_kernels == 4
    assert self.mlps[1].layer0.in_channels == 12 * 1
    assert self.mlps[1].layer0.out_channels == 32
    assert self.mlps[1].layer0.num_kernels == 8

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


def test_paconv_sa_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import build_sa_module
    sa_cfg = dict(
        type='PAConvSAModule',
        num_point=16,
        radius=0.2,
        num_sample=8,
        mlp_channels=[12, 32],
        paconv_num_kernels=[8],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max',
        paconv_kernel_input='w_neighbor')
    self = build_sa_module(sa_cfg).cuda()

    assert self.mlps[0].layer0.in_channels == 15 * 2
    assert self.mlps[0].layer0.out_channels == 32
    assert self.mlps[0].layer0.num_kernels == 8

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

    # test kNN sampling when radius is None
    sa_cfg = dict(
        type='PAConvSAModule',
        num_point=16,
        radius=None,
        num_sample=8,
        mlp_channels=[12, 32],
        paconv_num_kernels=[8],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max',
        paconv_kernel_input='identity')
    self = build_sa_module(sa_cfg).cuda()
    assert self.mlps[0].layer0.in_channels == 15 * 1

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).cuda()
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 32, 16])
    assert inds.shape == torch.Size([1, 16])


def test_paconv_cuda_sa_module_msg():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import PAConvCUDASAModuleMSG

    # paconv_num_kernels should have same length as mlp_channels
    with pytest.raises(AssertionError):
        self = PAConvCUDASAModuleMSG(
            num_point=16,
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            paconv_num_kernels=[[4]]).cuda()

    # paconv_num_kernels inner num should match as mlp_channels
    with pytest.raises(AssertionError):
        self = PAConvCUDASAModuleMSG(
            num_point=16,
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            paconv_num_kernels=[[4, 4], [8, 8]]).cuda()

    self = PAConvCUDASAModuleMSG(
        num_point=16,
        radii=[0.2, 0.4],
        sample_nums=[4, 8],
        mlp_channels=[[12, 16], [12, 32]],
        paconv_num_kernels=[[4], [8]],
        norm_cfg=dict(type='BN2d'),
        use_xyz=False,
        pool_mod='max',
        paconv_kernel_input='w_neighbor').cuda()

    assert self.mlps[0][0].in_channels == 12 * 2
    assert self.mlps[0][0].out_channels == 16
    assert self.mlps[0][0].num_kernels == 4
    assert self.mlps[0][0].bn.num_features == 16
    assert self.mlps[1][0].in_channels == 12 * 2
    assert self.mlps[1][0].out_channels == 32
    assert self.mlps[1][0].num_kernels == 8
    assert self.mlps[1][0].bn.num_features == 32

    assert self.mlps[0][0].scorenet.mlps.layer0.conv.in_channels == 7
    assert self.mlps[0][0].scorenet.mlps.layer3.conv.out_channels == 4
    assert self.mlps[1][0].scorenet.mlps.layer0.conv.in_channels == 7
    assert self.mlps[1][0].scorenet.mlps.layer3.conv.out_channels == 8

    # last conv in ScoreNet has neither bn nor relu
    with pytest.raises(AttributeError):
        _ = self.mlps[0][0].scorenet.mlps.layer3.bn
    with pytest.raises(AttributeError):
        _ = self.mlps[0][0].scorenet.mlps.layer3.activate

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

    # CUDA PAConv only supports w_neighbor kernel_input
    with pytest.raises(AssertionError):
        self = PAConvCUDASAModuleMSG(
            num_point=16,
            radii=[0.2, 0.4],
            sample_nums=[4, 8],
            mlp_channels=[[12, 16], [12, 32]],
            paconv_num_kernels=[[4], [8]],
            norm_cfg=dict(type='BN2d'),
            use_xyz=False,
            pool_mod='max',
            paconv_kernel_input='identity').cuda()


def test_paconv_cuda_sa_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.ops import build_sa_module
    sa_cfg = dict(
        type='PAConvCUDASAModule',
        num_point=16,
        radius=0.2,
        num_sample=8,
        mlp_channels=[12, 32],
        paconv_num_kernels=[8],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max',
        paconv_kernel_input='w_neighbor')
    self = build_sa_module(sa_cfg).cuda()

    assert self.mlps[0][0].in_channels == 15 * 2
    assert self.mlps[0][0].out_channels == 32
    assert self.mlps[0][0].num_kernels == 8

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

    # test kNN sampling when radius is None
    sa_cfg = dict(
        type='PAConvCUDASAModule',
        num_point=16,
        radius=None,
        num_sample=8,
        mlp_channels=[12, 32],
        paconv_num_kernels=[8],
        norm_cfg=dict(type='BN2d'),
        use_xyz=True,
        pool_mod='max',
        paconv_kernel_input='w_neighbor')
    self = build_sa_module(sa_cfg).cuda()

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).cuda()
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()
    new_xyz, new_features, inds = self(xyz, features)
    assert new_xyz.shape == torch.Size([1, 16, 3])
    assert new_features.shape == torch.Size([1, 32, 16])
    assert inds.shape == torch.Size([1, 16])
