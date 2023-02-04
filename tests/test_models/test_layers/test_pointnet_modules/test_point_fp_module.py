# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


def test_pointnet_fp_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.models.layers import PointFPModule

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
