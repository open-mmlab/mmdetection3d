# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from mmengine.device import is_musa_available, is_cuda_available

from mmdet3d.registry import MODELS


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
def test_spvcnn_backbone(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip('test requires GPU and torch+cuda')

    try:
        import torchsparse  # noqa: F401
    except ImportError:
        pytest.skip('test requires Torchsparse installation')

    coordinates, features = [], []
    for i in range(2):
        c = torch.randint(0, 10, (100, 3)).int()
        c = F.pad(c, (0, 1), mode='constant', value=i)
        coordinates.append(c)
        f = torch.rand(100, 4)
        features.append(f)
    features = torch.cat(features, dim=0).to(device)
    coordinates = torch.cat(coordinates, dim=0).to(device)

    cfg = dict(type='SPVCNNBackbone')
    self = MODELS.build(cfg).to(device)
    self.init_weights()

    y = self(features, coordinates)
    assert y.F.shape == torch.Size([200, 96])
    assert y.C.shape == torch.Size([200, 4])
