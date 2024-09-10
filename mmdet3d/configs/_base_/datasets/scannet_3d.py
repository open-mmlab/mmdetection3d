# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset.dataset_wrapper import RepeatDataset
from mmengine.dataset.sampler import DefaultSampler
from mmengine.visualization.vis_backend import LocalVisBackend

from mmdet3d.datasets.scannet_dataset import ScanNetDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile,
                                                 PointSegClassMapping)
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.transforms.transforms_3d import (GlobalAlignment,
                                                       GlobalRotScaleTrans,
                                                       PointSample,
                                                       RandomFlip3D)
from mmdet3d.evaluation.metrics.indoor_metric import IndoorMetric
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# dataset settings
dataset_type = 'ScanNetDataset'
data_root = 'data/scannet/'

metainfo = dict(
    classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
             'bookshelf', 'picture', 'counter', 'desk', 'curtain',
             'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
             'garbagebin'))

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/scannet/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type=GlobalAlignment, rotation_axis=2),
    dict(type=PointSegClassMapping),
    dict(type=PointSample, num_points=40000),
    dict(
        type=RandomFlip3D,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True),
    dict(
        type=Pack3DDetInputs,
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(type=GlobalAlignment, rotation_axis=2),
    dict(
        type=MultiScaleFlipAug3D,
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type=GlobalRotScaleTrans,
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type=RandomFlip3D,
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type=PointSample, num_points=40000),
        ]),
    dict(type=Pack3DDetInputs, keys=['points'])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=RepeatDataset,
        times=5,
        dataset=dict(
            type=ScanNetDataset,
            data_root=data_root,
            ann_file='scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ScanNetDataset,
        data_root=data_root,
        ann_file='scannet_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ScanNetDataset,
        data_root=data_root,
        ann_file='scannet_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=backend_args))
val_evaluator = dict(type=IndoorMetric)
test_evaluator = val_evaluator

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer')
