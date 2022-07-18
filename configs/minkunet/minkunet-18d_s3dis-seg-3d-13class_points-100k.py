_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py',
    '../_base_/models/minkunet.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
num_points = 100000
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')

# data settings
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
)
evaluation = dict(interval=1)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# model settings
model = dict(
    type='SparseEncoderDecoder3D',
    voxel_size=0.05,
    backbone=dict(
        type='MinkUNetBase',
        depth=18,
        in_channels=3,
        D=3,
    ),
    decode_head=dict(
        type='MinkUNetHead',
        num_classes=13,
        ignore_index=13,
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        )))

# runtime settings
checkpoint_config = dict(interval=5)
