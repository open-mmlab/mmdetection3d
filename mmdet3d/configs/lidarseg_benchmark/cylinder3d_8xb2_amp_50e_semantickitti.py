# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .cylinder3d_8xb2_50e_semantickitti import *

optim_wrapper.update(dict(type='AmpOptimWrapper', loss_scale='dynamic'))
