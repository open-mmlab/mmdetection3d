# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.ops import PAConv, PAConvCUDA


def test_paconv():
    B = 2
    in_channels = 6
    out_channels = 12
    npoint = 4
    K = 3
    num_kernels = 4
    points_xyz = torch.randn(B, 3, npoint, K)
    features = torch.randn(B, in_channels, npoint, K)

    paconv = PAConv(in_channels, out_channels, num_kernels)
    assert paconv.weight_bank.shape == torch.Size(
        [in_channels * 2, out_channels * num_kernels])

    with torch.no_grad():
        new_features, _ = paconv((features, points_xyz))

    assert new_features.shape == torch.Size([B, out_channels, npoint, K])


def test_paconv_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    B = 2
    in_channels = 6
    out_channels = 12
    N = 32
    npoint = 4
    K = 3
    num_kernels = 4
    points_xyz = torch.randn(B, 3, npoint, K).float().cuda()
    features = torch.randn(B, in_channels, N).float().cuda()
    points_idx = torch.randint(0, N, (B, npoint, K)).long().cuda()

    paconv = PAConvCUDA(in_channels, out_channels, num_kernels).cuda()
    assert paconv.weight_bank.shape == torch.Size(
        [in_channels * 2, out_channels * num_kernels])

    with torch.no_grad():
        new_features, _, _ = paconv((features, points_xyz, points_idx))

    assert new_features.shape == torch.Size([B, out_channels, npoint, K])
