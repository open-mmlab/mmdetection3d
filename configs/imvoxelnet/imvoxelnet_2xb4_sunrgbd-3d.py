_base_ = [
    '../_base_/datasets/sunrgbd-3d.py', '../_base_/models/imvoxelnet.py',
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]

# model settings
prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-3.2, -0.2, -2.28, 3.2, 6.2, 0.28]],
    rotations=[.0])
model = dict(
    neck=dict(out_channels=256),
    neck_3d=dict(
        _delete_=True,
        type='IndoorImVoxelNeck',
        in_channels=256,
        out_channels=128,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        _delete_=True,
        type='ImVoxelHead',
        n_classes=10,
        n_levels=3,
        n_channels=128,
        n_reg_outs=7,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        prior_generator=prior_generator),
    prior_generator=prior_generator,
    n_voxels=[40, 40, 16],
    coord_type='DEPTH',
    train_cfg=dict(_delete_=True),
    test_cfg=dict(_delete_=True, nms_pre=1000, iou_thr=.25, score_thr=.01))

# dataset settings
# dataset_type = 'SUNRGBDDataset'
# data_root = 'data/sunrgbd/'
# class_names = [
#     'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
#     'night_stand', 'bookshelf', 'bathtub'
# ]
# metainfo = dict(CLASSES=class_names)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='RandomResize', scale=[(512, 384), (768, 576)], keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 480), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        times=2, dataset=dict(pipeline=train_pipeline, filter_empty_gt=True)))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

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

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)
