# Copyright (c) OpenMMLab. All rights reserved.

import torch
import pytest
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
def test_dla_neck(device: str):

    s = 32
    in_channels = [16, 32, 64, 128, 256, 512]
    feat_sizes = [s // 2**i for i in range(6)]  # [32, 16, 8, 4, 2, 1]

    if is_cuda_available() or is_musa_available():
        # Test DLA Neck with DCNv2 on GPU
        neck_cfg = dict(
            type='DLANeck',
            in_channels=[16, 32, 64, 128, 256, 512],
            start_level=2,
            end_level=5,
            norm_cfg=dict(type='GN', num_groups=32))
        neck = MODELS.build(neck_cfg)
        neck.init_weights()
        neck.to(device)
        feats = [
            torch.rand(4, in_channels[i], feat_sizes[i], feat_sizes[i]).to(device)
            for i in range(len(in_channels))
        ]
        outputs = neck(feats)
        assert outputs[0].shape == (4, 64, 8, 8)
    else:
        # Test DLA Neck without DCNv2 on CPU
        neck_cfg = dict(
            type='DLANeck',
            in_channels=[16, 32, 64, 128, 256, 512],
            start_level=2,
            end_level=5,
            norm_cfg=dict(type='GN', num_groups=32),
            use_dcn=False)
        neck = MODELS.build(neck_cfg)
        neck.init_weights()
        feats = [
            torch.rand(4, in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(in_channels))
        ]
        outputs = neck(feats)
        assert outputs[0].shape == (4, 64, 8, 8)
