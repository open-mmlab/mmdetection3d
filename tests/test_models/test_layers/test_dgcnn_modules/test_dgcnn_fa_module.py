# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


def test_dgcnn_fa_module():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.models.layers import DGCNNFAModule

    self = DGCNNFAModule(mlp_channels=[24, 16]).cuda()
    assert self.mlps.layer0.conv.in_channels == 24
    assert self.mlps.layer0.conv.out_channels == 16

    points = [torch.rand(1, 200, 12).float().cuda() for _ in range(3)]

    fa_points = self(points)
    assert fa_points.shape == torch.Size([1, 200, 40])
