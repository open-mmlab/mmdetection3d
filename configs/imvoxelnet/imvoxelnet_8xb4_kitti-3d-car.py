_base_ = [
    '../_base_/datasets/kitti-3d-car.py', '../_base_/models/imvoxelnet.py',
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]

input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
data_prefix = dict(img='training/image_2')

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='RandomResize', scale=[(1173, 352), (1387, 416)],
        keep_ratio=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        times=3,
        dataset=dict(
            data_prefix=data_prefix,
            pipeline=train_pipeline,
            modality=input_modality)))
val_dataloader = dict(
    dataset=dict(
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality))
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
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
