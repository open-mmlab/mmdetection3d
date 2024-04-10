# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.semantickitti import *
    from .._base_.models.spvcnn import *
    from .._base_.schedules.lidarseg_50e import *
    from .._base_.default_runtime import *

from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend

model.update(dict(data_preprocessor=dict(max_voxels=None)))

visualizer.update(
    dict(vis_backends=[dict(type=LocalVisBackend),
                       dict(type=WandbVisBackend)]))
default_hooks.update(
    dict(checkpoint=dict(type=CheckpointHook, save_best='miou')))
