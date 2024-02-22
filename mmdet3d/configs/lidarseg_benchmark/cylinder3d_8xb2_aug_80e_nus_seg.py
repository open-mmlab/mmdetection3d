# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .cylinder3d_8xb2_80e_nus_seg import *

from mmcv.transforms.wrappers import RandomChoice

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile,
                                                 PointSegClassMapping)
from mmdet3d.datasets.transforms.transforms_3d import (GlobalRotScaleTrans,
                                                       LaserMix, PolarMix)

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args),
    dict(type=PointSegClassMapping),
    dict(
        type=RandomChoice,
        transforms=[
            [
                dict(
                    type=LaserMix,
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-30, 10],
                    pre_transform=[
                        dict(
                            type=LoadPointsFromFile,
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=4,
                            backend_args=backend_args),
                        dict(
                            type=LoadAnnotations3D,
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.uint8',
                            backend_args=backend_args),
                        dict(type=PointSegClassMapping)
                    ],
                    prob=1)
            ],
            [
                dict(
                    type=PolarMix,
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type=LoadPointsFromFile,
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=4,
                            backend_args=backend_args),
                        dict(
                            type=LoadAnnotations3D,
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.uint8',
                            backend_args=backend_args),
                        dict(type=PointSegClassMapping)
                    ],
                    prob=1)
            ],
        ],
        prob=[0.5, 0.5]),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type=Pack3DDetInputs, keys=['points', 'pts_semantic_mask'])
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))
