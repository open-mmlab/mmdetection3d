# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.nus_mono3d import *
    from .._base_.models.fcos3d import *
    from .._base_.schedules.mmdet_schedule_1x import *
    from .._base_.default_runtime import *

from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCNv2
from mmcv.transforms.processing import Resize

from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadImageFromFileMono3D,
                                                 LoadAnnotations3D)
from mmdet3d.datasets.transforms.transforms_3d import RandomFlip3D
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

# model settings
model.update(
    dict(
        data_preprocessor=dict(
            type=Det3DDataPreprocessor,
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32),
        backbone=dict(
            dcn=dict(type=DCNv2, deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True))))

backend_args = None

train_pipeline = [
    dict(type=LoadImageFromFileMono3D, backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type=Resize, scale=(1600, 900), keep_ratio=True),
    dict(type=RandomFlip3D, flip_ratio_bev_horizontal=0.5),
    dict(
        type=Pack3DDetInputs,
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]

test_pipeline = [
    dict(type=LoadImageFromFileMono3D, backend_args=backend_args),
    dict(type=Resize, scale_factor=1.0),
    dict(type=Pack3DDetInputs, keys=['img'])
]

train_dataloader.update(
    dict(batch_size=2, num_workers=2, dataset=dict(pipeline=train_pipeline)))
test_dataloader.update(dict(dataset=dict(pipeline=test_pipeline)))
val_dataloader.update(dict(dataset=dict(pipeline=test_pipeline)))

# optimizer
optim_wrapper.update(
    dict(
        optimizer=dict(lr=0.002),
        paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
        clip_grad=dict(max_norm=35, norm_type=2)))

# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
