# Copyright (c) OpenMMLab. All rights reserved.
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
def test_dgcnn_fa_module(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()
    from mmdet3d.models.layers import DGCNNFAModule

    self = DGCNNFAModule(mlp_channels=[24, 16]).to(device)
    assert self.mlps.layer0.conv.in_channels == 24
    assert self.mlps.layer0.conv.out_channels == 16

    points = [torch.rand(1, 200, 12).float().to(device) for _ in range(3)]

    fa_points = self(points)
    assert fa_points.shape == torch.Size([1, 200, 40])
