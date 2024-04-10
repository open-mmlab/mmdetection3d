# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .minkunet18_w32_spconv_8xb2_amp_80e_nus_seg import *

model.update(
    dict(
        backbone=dict(
            base_channels=20,
            encoder_channels=[20, 40, 81, 163],
            decoder_channels=[163, 81, 61, 61]),
        decode_head=dict(channels=61)))
