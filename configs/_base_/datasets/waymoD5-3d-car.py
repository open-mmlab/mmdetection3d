# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://waymo_data/'))

class_names = ['Car']
metainfo = dict(classes=class_names)

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4]))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='waymo_infos_train.pkl',
            data_prefix=dict(
                pts='training/velodyne', sweeps='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=5)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/gt.bin',
    data_root='./data/waymo/waymo_format',
    convert_kitti_format=False)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
