# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


def test_dgcnn_fp_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.models.layers import DGCNNFPModule

    self = DGCNNFPModule(mlp_channels=[24, 16]).cuda()
    assert self.mlps.layer0.conv.in_channels == 24
    assert self.mlps.layer0.conv.out_channels == 16

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin',
                      np.float32).reshape((-1, 6))

    # (B, N, 3)
    xyz = torch.from_numpy(xyz).view(1, -1, 3).cuda()
    points = xyz.repeat([1, 1, 8]).cuda()

    fp_points = self(points)
    assert fp_points.shape == torch.Size([1, 200, 16])
