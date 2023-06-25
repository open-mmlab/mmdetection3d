# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset.sampler import DefaultSampler
from mmengine.visualization.vis_backend import LocalVisBackend

from mmdet3d.datasets.lyft_dataset import LyftDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile,
                                                 LoadPointsFromMultiSweeps)
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.transforms.transforms_3d import (GlobalRotScaleTrans,
                                                       ObjectRangeFilter,
                                                       PointShuffle,
                                                       PointsRangeFilter,
                                                       RandomFlip3D)
from mmdet3d.evaluation.metrics.lyft_metric import LyftMetric
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-100, -100, -5, 100, 100, 3]
# For Lyft we usually do 9-class detection
class_names = [
    'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 'motorcycle',
    'bicycle', 'pedestrian', 'animal'
]
dataset_type = 'LyftDataset'
data_root = 'data/lyft/'
data_prefix = dict(pts='v1.01-train/lidar', img='', sweeps='v1.01-train/lidar')
# Input modality for Lyft dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/lyft/'

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
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=10,
        backend_args=backend_args),
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type=RandomFlip3D, flip_ratio_bev_horizontal=0.5),
    dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=PointShuffle),
    dict(
        type=Pack3DDetInputs, keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=10,
        backend_args=backend_args),
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
            dict(type=RandomFlip3D),
            dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
        ]),
    dict(type=Pack3DDetInputs, keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=10,
        backend_args=backend_args),
    dict(type=Pack3DDetInputs, keys=['points'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=LyftDataset,
        data_root=data_root,
        ann_file='lyft_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=False,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=LyftDataset,
        data_root=data_root,
        ann_file='lyft_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=LyftMetric,
    data_root=data_root,
    ann_file='lyft_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer')
