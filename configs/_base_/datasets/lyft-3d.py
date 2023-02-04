# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-80, -80, -5, 80, 80, 3]
# For Lyft we usually do 9-class detection
class_names = [
    'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 'motorcycle',
    'bicycle', 'pedestrian', 'animal'
]
dataset_type = 'LyftDataset'
data_root = 'data/lyft/'
# Input modality for Lyft dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/lyft/': 's3://lyft/lyft/',
#         'data/lyft/': 's3://lyft/lyft/'
#    }))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
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
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='lyft_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=False,
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
        ann_file='lyft_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR'))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='lyft_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR'))

val_evaluator = dict(
    type='LyftMetric',
    ann_file=data_root + 'lyft_infos_val.pkl',
    metric='bbox')
test_evaluator = dict(
    type='LyftMetric',
    ann_file=data_root + 'lyft_infos_val.pkl',
    metric='bbox')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
