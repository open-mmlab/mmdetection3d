import pytest
import torch

from mmdet3d.models.builder import build_neck


def test_imvoxel_neck():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    neck_cfg = dict(
        type='OutdoorImVoxelNeck', in_channels=64, out_channels=256)
    neck = build_neck(neck_cfg).cuda()
    inputs = torch.rand([1, 64, 216, 248, 12], device='cuda')
    outputs = neck(inputs)
    assert outputs[0].shape == (1, 256, 248, 216)
