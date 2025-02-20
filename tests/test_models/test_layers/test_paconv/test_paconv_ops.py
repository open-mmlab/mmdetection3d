# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.device import is_musa_available, is_cuda_available

from mmdet3d.models.layers import PAConv, PAConvCUDA


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
def test_paconv(device: str):
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
def test_paconv_cuda(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()
    B = 2
    in_channels = 6
    out_channels = 12
    N = 32
    npoint = 4
    K = 3
    num_kernels = 4
    points_xyz = torch.randn(B, 3, npoint, K).float().to(device)
    features = torch.randn(B, in_channels, N).float().to(device)
    points_idx = torch.randint(0, N, (B, npoint, K)).long().to(device)

    paconv = PAConvCUDA(in_channels, out_channels, num_kernels).to(device)
    assert paconv.weight_bank.shape == torch.Size(
        [in_channels * 2, out_channels * num_kernels])

    with torch.no_grad():
        new_features, _, _ = paconv((features, points_xyz, points_idx))

    assert new_features.shape == torch.Size([B, out_channels, npoint, K])
