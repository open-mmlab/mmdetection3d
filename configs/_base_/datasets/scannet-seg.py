# For ScanNet seg we usually do 20-class segmentation
class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')
metainfo = dict(classes=class_names)
dataset_type = 'ScanNetSegDataset'
data_root = 'data/scannet/'
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/scannet/':
#         's3://scannet/',
#     }))

num_points = 8192

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/',
#         'data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/'
#     }))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
<<<<<<< HEAD:configs/_base_/datasets/scannet_seg-3d-20class.py
        with_seg_3d=True,
        file_client_args=file_client_args),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39),
        max_cat_id=40),
=======
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-seg.py
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.5,
        ignore_index=len(class_names),
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
<<<<<<< HEAD:configs/_base_/datasets/scannet_seg-3d-20class.py
        use_dim=[0, 1, 2, 3, 4, 5],
        file_client_args=file_client_args),
=======
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True),
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-seg.py
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        # a wrapper in order to successfully call test function
        # actually we don't perform test-time-aug
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
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
<<<<<<< HEAD:configs/_base_/datasets/scannet_seg-3d-20class.py
        use_dim=[0, 1, 2, 3, 4, 5],
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=file_client_args),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39),
        max_cat_id=40),
    dict(
        type='DefaultFormatBundle3D',
        with_label=False,
        class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
=======
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-seg.py
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_train.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=data_root + 'seg_info/train_resampled_scene_idxs.npy',
<<<<<<< HEAD:configs/_base_/datasets/scannet_seg-3d-20class.py
        file_client_args=file_client_args),
    val=dict(
=======
        test_mode=False))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-seg.py
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
<<<<<<< HEAD:configs/_base_/datasets/scannet_seg-3d-20class.py
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names),
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names),
        file_client_args=file_client_args))
=======
        modality=input_modality,
        ignore_index=len(class_names),
        test_mode=True))
val_dataloader = test_dataloader
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973:configs/_base_/datasets/scannet-seg.py

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
