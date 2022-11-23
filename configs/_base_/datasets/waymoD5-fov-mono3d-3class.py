# dataset settings
# D3 in the config name means the whole dataset is divided into 3 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    # base shape (1248, 832), scale (0.95, 1.05)
    dict(
        type='RandomResize3D',
        scale=(1284, 832),
        ratio_range=(0.95, 1.05),
        keep_ratio=True,
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='RandomResize3D',
        scale=(1248, 832),
        ratio_range=(1., 1.),
        keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='RandomResize3D',
        scale=(1248, 832),
        ratio_range=(1., 1.),
        keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img']),
]

metainfo = dict(CLASSES=class_names)

train_dataloader = dict(
    batch_size=3,
    num_workers=3,
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
            CAM_SIDE_LEFT='training/image_4'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='fov_image_based',
        # load one frame every three frames
        load_interval=5))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='fov_image_based',
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
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4'),
        ann_file='waymo_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Camera',
        load_type='fov_image_based',
    ))

val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/fov_gt.bin',
    data_root='./data/waymo/waymo_format',
    metric='LET_mAP',
    load_type='fov_image_based',
)
test_evaluator = val_evaluator
