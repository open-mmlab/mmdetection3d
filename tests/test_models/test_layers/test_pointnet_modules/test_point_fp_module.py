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
def test_pointnet_fp_module(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()
    from mmdet3d.models.layers import PointFPModule

    self = PointFPModule(mlp_channels=[24, 16]).to(device)
    assert self.mlps.layer0.conv.in_channels == 24
    assert self.mlps.layer0.conv.out_channels == 16

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin',
                      np.float32).reshape((-1, 6))

    # (B, N, 3)
    xyz1 = torch.from_numpy(xyz[0::2, :3]).view(1, -1, 3).to(device)
    # (B, C1, N)
    features1 = xyz1.repeat([1, 1, 4]).transpose(1, 2).contiguous().to(device)

    # (B, M, 3)
    xyz2 = torch.from_numpy(xyz[1::3, :3]).view(1, -1, 3).to(device)
    # (B, C2, N)
    features2 = xyz2.repeat([1, 1, 4]).transpose(1, 2).contiguous().to(device)

    fp_features = self(xyz1, xyz2, features1, features2)
    assert fp_features.shape == torch.Size([1, 16, 50])
