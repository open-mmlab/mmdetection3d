# dataset settings
dataset_type = 'ScanNetDataset'
<<<<<<< HEAD:configs/_base_/datasets/scannet-3d-18class.py
data_root = './data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

file_client_args = dict(backend='disk')
=======
data_root = 'data/scannet/'

metainfo = dict(
    classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
             'bookshelf', 'picture', 'counter', 'desk', 'curtain',
             'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
             'garbagebin'))

# file_client_args = dict(backend='disk')
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-3d.py
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/scannet/':
<<<<<<< HEAD:configs/_base_/datasets/scannet-3d-18class.py
#         's3://openmmlab/datasets/detection3d/scannet_processed/',
#         'data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/'
=======
#         's3://scannet/',
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-3d.py
#     }))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        file_client_args=file_client_args,
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        file_client_args=file_client_args,
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSegClassMapping'),
    dict(type='PointSample', num_points=40000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        file_client_args=file_client_args,
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
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
                flip_ratio_bev_vertical=0.5),
            dict(type='PointSample', num_points=40000),
<<<<<<< HEAD:configs/_base_/datasets/scannet-3d-18class.py
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
        file_client_args=file_client_args,
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
=======
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-3d.py
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
<<<<<<< HEAD:configs/_base_/datasets/scannet-3d-18class.py
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
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-3d.py
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val.pkl',
        pipeline=test_pipeline,
<<<<<<< HEAD:configs/_base_/datasets/scannet-3d-18class.py
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
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
        ann_file='scannet_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth'))
val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-3d.py

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
