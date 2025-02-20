import pytest
import torch
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
def test_imvoxel_neck(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip('test requires GPU and torch+cuda')

    neck_cfg = dict(
        type='OutdoorImVoxelNeck', in_channels=64, out_channels=256)
    neck = MODELS.build(neck_cfg).to(device)
    inputs = torch.rand([1, 64, 216, 248, 12], device=device)
    outputs = neck(inputs)
    assert outputs[0].shape == (1, 256, 248, 216)
