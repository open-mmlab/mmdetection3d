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
def test_dgcnn_fp_module(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()
    from mmdet3d.models.layers import DGCNNFPModule

    self = DGCNNFPModule(mlp_channels=[24, 16]).to(device)
    assert self.mlps.layer0.conv.in_channels == 24
    assert self.mlps.layer0.conv.out_channels == 16

    xyz = np.fromfile("tests/data/sunrgbd/points/000001.bin", np.float32).reshape(
        (-1, 6)
    )

    # (B, N, 3)
    xyz = torch.from_numpy(xyz).view(1, -1, 3).to(device)
    points = xyz.repeat([1, 1, 8]).to(device)

    fp_points = self(points)
    assert fp_points.shape == torch.Size([1, 200, 16])
