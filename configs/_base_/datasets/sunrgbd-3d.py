dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

<<<<<<< HEAD:configs/_base_/datasets/sunrgbd-3d-10class.py
=======
metainfo = dict(classes=class_names)

>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/sunrgbd-3d.py
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/sunrgbd/':
<<<<<<< HEAD:configs/_base_/datasets/sunrgbd-3d-10class.py
#         's3://openmmlab/datasets/detection3d/sunrgbd_processed/',
#         'data/sunrgbd/':
#         's3://openmmlab/datasets/detection3d/sunrgbd_processed/'
=======
#         's3://sunrgbd/',
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/sunrgbd-3d.py
#     }))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', file_client_args=file_client_args),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
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
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
<<<<<<< HEAD:configs/_base_/datasets/sunrgbd-3d-10class.py
            dict(type='PointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
=======
            dict(type='PointSample', num_points=20000)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/sunrgbd-3d.py
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
<<<<<<< HEAD:configs/_base_/datasets/sunrgbd-3d-10class.py
            box_type_3d='Depth',
            file_client_args=file_client_args)),
    val=dict(
=======
            box_type_3d='Depth')))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/sunrgbd-3d.py
        type=dataset_type,
        data_root=data_root,
        ann_file='sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
<<<<<<< HEAD:configs/_base_/datasets/sunrgbd-3d-10class.py
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args))
=======
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth'))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth'))
val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/sunrgbd-3d.py

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
