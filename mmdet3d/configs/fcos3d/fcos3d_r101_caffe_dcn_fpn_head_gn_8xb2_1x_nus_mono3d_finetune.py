# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .fcos3d_r101_caffe_dcn_fpn_head_gn_8xb2_1x_nus_mono3d import *

# model settings
model.update(
    dict(
        train_cfg=dict(
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05])))

# optimizer
optim_wrapper.update(dict(optimizer=dict(lr=0.001)))
load_from = 'work_dirs/fcos3d_nus/latest.pth'
