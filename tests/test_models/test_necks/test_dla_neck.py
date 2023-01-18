# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdet3d.registry import MODELS


def test_dla_neck():

    s = 32
    in_channels = [16, 32, 64, 128, 256, 512]
    feat_sizes = [s // 2**i for i in range(6)]  # [32, 16, 8, 4, 2, 1]

    if torch.cuda.is_available():
        # Test DLA Neck with DCNv2 on GPU
        neck_cfg = dict(
            type='DLANeck',
            in_channels=[16, 32, 64, 128, 256, 512],
            start_level=2,
            end_level=5,
            norm_cfg=dict(type='GN', num_groups=32))
        neck = MODELS.build(neck_cfg)
        neck.init_weights()
        neck.cuda()
        feats = [
            torch.rand(4, in_channels[i], feat_sizes[i], feat_sizes[i]).cuda()
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
