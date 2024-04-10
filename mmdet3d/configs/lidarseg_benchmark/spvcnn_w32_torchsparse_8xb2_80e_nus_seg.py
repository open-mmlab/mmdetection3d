# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.nus_seg import *
    from .._base_.models.spvcnn import *
    from .._base_.schedules.lidarseg_80e import *
    from .._base_.default_runtime import *

from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend

model.update(
    dict(
        data_preprocessor=dict(
            max_voxels=None, voxel_layer=dict(voxel_size=[0.1, 0.1, 0.1])),
        decode_head=dict(num_classes=17, ignore_index=16)))

train_dataloader.update(
    dict(dataset=dict(ann_file='nuscenes_lidarseg_infos_train.pkl')))
test_dataloader.update(
    dict(dataset=dict(ann_file='nuscenes_lidarseg_infos_val.pkl')))
visualizer.update(
    dict(vis_backends=[dict(type=LocalVisBackend),
                       dict(type=WandbVisBackend)]))
default_hooks.update(
    dict(checkpoint=dict(type=CheckpointHook, save_best='miou')))
