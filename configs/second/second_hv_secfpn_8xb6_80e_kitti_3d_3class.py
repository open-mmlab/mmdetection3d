# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.kitti_3d_3class import *
    from .._base_.default_runtime import *
    from .._base_.models.second_hv_secfpn_kitti import *
    from .._base_.schedules.cyclic_40e import *
