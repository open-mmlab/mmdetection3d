_base_ = [
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImVoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_3d=dict(type='OutdoorImVoxelNeck', in_channels=64, out_channels=256),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-0.16, -39.68, -1.78, 68.96, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    n_voxels=[216, 248, 12],
    anchor_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[-0.16, -39.68, -3.08, 68.96, 39.68, 0.76]],
        rotations=[.0]),
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
metainfo = dict(classes=class_names)

# file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/kitti/':
        's3://openmmlab/datasets/detection3d/kitti/',
        'data/kitti/':
        's3://openmmlab/datasets/detection3d/kitti/'
    }))

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='LoadImageFromFileMono3D'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='RandomResize', scale=[(1173, 352), (1387, 416)],
        keep_ratio=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(img='training/image_2'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', max_keep_ckpts=1))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
