# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .minkunet34_w32_torchsparse2_8xb2_aug_50e_semantickitti import *

optim_wrapper.update(dict(type='AmpOptimWrapper', loss_scale='dynamic'))
