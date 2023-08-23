# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from ..second.second_hv_secfpn_8xb6_80e_kitti_3d_3class import *

from mmengine.optim.optimizer import AmpOptimWrapper

# schedule settings
optim_wrapper.update(type=AmpOptimWrapper, loss_scale=4096.)
