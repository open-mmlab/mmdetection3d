_base_ = [
    '../_base_/models/fcaf3d.py', '../_base_/default_runtime.py',
    '../_base_/datasets/sunrgbd-3d.py'
]

n_points = 100000
backend_args = None

model = dict(
    bbox_head=dict(
        num_classes=10,
        num_reg_outs=8,
        bbox_loss=dict(type='RotatedIoU3DLoss')))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(type='PointSample', num_points=n_points),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        times=3, dataset=dict(pipeline=train_pipeline, filter_empty_gt=True)))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))

# learning rate
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=12,
    by_epoch=True,
    milestones=[8, 11],
    gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
