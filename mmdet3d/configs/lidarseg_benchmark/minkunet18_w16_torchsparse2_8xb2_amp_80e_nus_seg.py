# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .minkunet18_w32_spconv_8xb2_amp_80e_nus_seg import *

model.update(
    dict(
        backbone=dict(
            base_channels=16,
            encoder_channels=[16, 32, 64, 128],
            decoder_channels=[128, 64, 48, 48]),
        decode_head=dict(channels=48)))
