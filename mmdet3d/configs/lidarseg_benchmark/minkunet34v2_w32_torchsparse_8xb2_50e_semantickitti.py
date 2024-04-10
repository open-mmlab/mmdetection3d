# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .minkunet34_w32_torchsparse2_8xb2_50e_semantickitti import *

model.update(
    dict(
        data_preprocessor=dict(batch_first=False),
        backbone=dict(type='MinkUNetBackboneV2'),
        decode_head=dict(channels=256 + 128 + 96)))
