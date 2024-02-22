# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .minkunet18_w32_torchsparse2_8xb2_amp_50e_semantickitti import *

model.update(
    dict(
        backbone=dict(
            base_channels=16,
            encoder_channels=[16, 32, 64, 128],
            decoder_channels=[128, 64, 48, 48]),
        decode_head=dict(channels=48)))
