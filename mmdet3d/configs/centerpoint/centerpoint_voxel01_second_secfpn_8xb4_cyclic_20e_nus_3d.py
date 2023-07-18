# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.nus_3d import *
    from .._base_.models.centerpoint_voxel01_second_secfpn_nus import *
    from .._base_.schedules.cyclic_20e import *
    from .._base_.default_runtime import *

from mmengine.dataset.sampler import DefaultSampler

from mmdet3d.datasets.dataset_wrappers import CBGSDataset
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile,
                                                 LoadPointsFromMultiSweeps)
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.transforms.transforms_3d import (  # noqa
    GlobalRotScaleTrans, ObjectNameFilter, ObjectRangeFilter, ObjectSample,
    PointShuffle, PointsRangeFilter, RandomFlip3D)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_prefix.update(
    dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'))
model.update(
    dict(
        data_preprocessor=dict(
            voxel_layer=dict(point_cloud_range=point_cloud_range)),
        pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
        # model training and testing settings
        train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
        test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2]))))

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True),
    dict(type=ObjectSample, db_sampler=db_sampler),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type=RandomFlip3D,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectNameFilter, classes=class_names),
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
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
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
            dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range)
        ]),
    dict(type=Pack3DDetInputs, keys=['points'])
]

train_dataloader.merge(
    dict(
        _delete_=True,
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type=DefaultSampler, shuffle=True),
        dataset=dict(
            type=CBGSDataset,
            dataset=dict(
                type=NuScenesDataset,
                data_root=data_root,
                ann_file='nuscenes_infos_train.pkl',
                pipeline=train_pipeline,
                metainfo=dict(classes=class_names),
                test_mode=False,
                data_prefix=data_prefix,
                use_valid_flag=True,
                # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
                # and box_type_3d='Depth' in sunrgbd and scannet dataset.
                box_type_3d='LiDAR',
                backend_args=backend_args))))
test_dataloader.update(
    dict(
        dataset=dict(
            pipeline=test_pipeline, metainfo=dict(classes=class_names))))
val_dataloader.update(
    dict(
        dataset=dict(
            pipeline=test_pipeline, metainfo=dict(classes=class_names))))

train_cfg.update(dict(val_interval=20))
