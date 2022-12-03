# dataset settings
# D3 in the config name means the whole dataset is divided into 3 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.

class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [-35.0, -75.0, -2, 75.0, 75.0, 4]

train_transforms = [
    dict(type='PhotoMetricDistortion3D'),
    dict(
        type='RandomResize3D',
        scale=(1248, 832),
        ratio_range=(0.95, 1.05),
        keep_ratio=True),
    dict(type='RandomCrop3D', crop_size=(720, 1080)),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_box3d=False),
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='Pack3DDetInputs', keys=[
            'img',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ]),
]
test_transforms = [
    dict(
        type='RandomResize3D',
        scale=(1248, 832),
        ratio_range=(1., 1.),
        keep_ratio=True)
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]
metainfo = dict(classes=class_names)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4',
        ),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='Lidar',
        load_interval=5,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4',
        ),
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='Lidar',
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4',
        ),
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='Lidar',
    ))
val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/cam_gt.bin',
    data_root='./data/waymo/waymo_format',
    metric='LET_mAP')

test_evaluator = val_evaluator
