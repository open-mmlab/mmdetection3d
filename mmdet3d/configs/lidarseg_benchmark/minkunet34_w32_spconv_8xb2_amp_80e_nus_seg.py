# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .minkunet34_w32_torchsparse2_8xb2_amp_80e_nus_seg import *

model.update(
    dict(
        data_preprocessor=dict(batch_first=True),
        backbone=dict(sparseconv_backend='spconv')))
