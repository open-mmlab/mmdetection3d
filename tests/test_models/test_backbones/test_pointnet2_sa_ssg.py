# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.registry import MODELS


def test_pointnet2_sa_ssg():
    if not torch.cuda.is_available():
        pytest.skip()

    cfg = dict(
        type='PointNet2SASSG',
        in_channels=6,
        num_points=(32, 16),
        radius=(0.8, 1.2),
        num_samples=(16, 8),
        sa_channels=((8, 16), (16, 16)),
        fp_channels=((16, 16), (16, 16)))
    self = MODELS.build(cfg)
    self.cuda()
    assert self.SA_modules[0].mlps[0].layer0.conv.in_channels == 6
    assert self.SA_modules[0].mlps[0].layer0.conv.out_channels == 8
    assert self.SA_modules[0].mlps[0].layer1.conv.out_channels == 16
    assert self.SA_modules[1].mlps[0].layer1.conv.out_channels == 16
    assert self.FP_modules[0].mlps.layer0.conv.in_channels == 32
    assert self.FP_modules[0].mlps.layer0.conv.out_channels == 16
    assert self.FP_modules[1].mlps.layer0.conv.in_channels == 19

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz)
    fp_xyz = ret_dict['fp_xyz']
    fp_features = ret_dict['fp_features']
    fp_indices = ret_dict['fp_indices']
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_xyz[0].shape == torch.Size([1, 16, 3])
    assert fp_xyz[1].shape == torch.Size([1, 32, 3])
    assert fp_xyz[2].shape == torch.Size([1, 100, 3])
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert fp_indices[0].shape == torch.Size([1, 16])
    assert fp_indices[1].shape == torch.Size([1, 32])
    assert fp_indices[2].shape == torch.Size([1, 100])
    assert sa_xyz[0].shape == torch.Size([1, 100, 3])
    assert sa_xyz[1].shape == torch.Size([1, 32, 3])
    assert sa_xyz[2].shape == torch.Size([1, 16, 3])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])
    assert sa_indices[0].shape == torch.Size([1, 100])
    assert sa_indices[1].shape == torch.Size([1, 32])
    assert sa_indices[2].shape == torch.Size([1, 16])

    # test only xyz input without features
    cfg['in_channels'] = 3
    self = MODELS.build(cfg)
    self.cuda()
    ret_dict = self(xyz[..., :3])
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])
