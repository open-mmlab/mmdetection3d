# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset.sampler import DefaultSampler

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadImageFromFileMono3D)
from mmdet3d.datasets.transforms.transforms_3d import (RandomFlip3D,
                                                       RandomResize3D)
from mmdet3d.datasets.waymo_dataset import WaymoDataset
from mmdet3d.evaluation.metrics.waymo_metric import WaymoMetric

# dataset settings
# D3 in the config name means the whole dataset is divided into 3 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/waymo/kitti_format/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(type=LoadImageFromFileMono3D, backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    # base shape (1248, 832), scale (0.95, 1.05)
    dict(
        type=RandomResize3D,
        scale=(1284, 832),
        ratio_range=(0.95, 1.05),
        keep_ratio=True,
    ),
    dict(type=RandomFlip3D, flip_ratio_bev_horizontal=0.5),
    dict(
        type=Pack3DDetInputs,
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]

test_pipeline = [
    dict(type=LoadImageFromFileMono3D, backend_args=backend_args),
    dict(
        type=RandomResize3D,
        scale=(1248, 832),
        ratio_range=(1., 1.),
        keep_ratio=True),
    dict(type=Pack3DDetInputs, keys=['img']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type=LoadImageFromFileMono3D, backend_args=backend_args),
    dict(
        type=RandomResize3D,
        scale=(1248, 832),
        ratio_range=(1., 1.),
        keep_ratio=True),
    dict(type=Pack3DDetInputs, keys=['img']),
]

metainfo = dict(classes=class_names)

train_dataloader = dict(
    batch_size=3,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=WaymoDataset,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='mv_image_based',
        # load one frame every three frames
        load_interval=5,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=WaymoDataset,
        data_root=data_root,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='mv_image_based',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=WaymoDataset,
        data_root=data_root,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='mv_image_based',
        backend_args=backend_args))

val_evaluator = dict(
    type=WaymoMetric,
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/cam_gt.bin',
    data_root='./data/waymo/waymo_format',
    metric='LET_mAP',
    load_type='mv_image_based',
    backend_args=backend_args)
test_evaluator = val_evaluator
