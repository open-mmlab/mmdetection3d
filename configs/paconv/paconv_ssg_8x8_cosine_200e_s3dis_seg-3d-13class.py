_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py',
    '../_base_/models/paconv_ssg.py', '../_base_/default_runtime.py'
]

# data settings
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
num_points = 4096
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=tuple(range(len(class_names))),
        max_cat_id=13),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.0,
        ignore_index=len(class_names),
        use_normalized_coord=True,
        num_try=100000,
        enlarge_size=0.0,
        min_unique_num=num_points // 4,
        eps=0.0),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.141592653589793, 3.141592653589793],  # [-pi, pi]
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomJitterPoints',
        jitter_std=[0.01, 0.01, 0.01],
        clip_range=[-0.05, 0.05]),
    dict(type='RandomDropPointsColor', drop_ratio=0.2),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]

data = dict(samples_per_gpu=8, train=dict(pipeline=train_pipeline))
evaluation = dict(interval=1)

# model settings
model = dict(
    decode_head=dict(
        num_classes=13, ignore_index=13,
        loss_decode=dict(class_weight=None)),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=12))

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(port=29502)

# optimizer
lr = 0.2
optimizer = dict(type='SGD', lr=lr, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=lr * 0.01)
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
