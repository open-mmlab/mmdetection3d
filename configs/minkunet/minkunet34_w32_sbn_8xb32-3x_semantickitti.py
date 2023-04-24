_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/minkunet.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        encoder_blocks=[2, 3, 4, 6],
        decoder_blocks=[2, 2, 2, 2],
        norm_cfg=dict(type='TorchSparseSyncBN'),
    ))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping'),
    # dict(
    #     type='LaserMix',
    #     num_areas=[3, 4, 5, 6],
    #     pitch_angles=[-25, 3],
    #     pre_transform=[
    #         dict(
    #             type='LoadPointsFromFile',
    #             coord_type='LIDAR',
    #             load_dim=4,
    #             use_dim=4),
    #         dict(
    #             type='LoadAnnotations3D',
    #             with_bbox_3d=False,
    #             with_label_3d=False,
    #             with_seg_3d=True,
    #             seg_3d_dtype='np.int32',
    #             seg_offset=2**16,
    #             dataset_type='semantickitti'),
    #         dict(type='PointSegClassMapping')
    #     ],
    #     prob=0.5),
    # dict(
    #     type='PolarMix',
    #     instance_classes=[1, 2, 3, 4, 5, 6, 7, 8],
    #     swap_ratio=0.5,
    #     rotate_paste_ratio=1.0,
    #     pre_transform=[
    #         dict(
    #             type='LoadPointsFromFile',
    #             coord_type='LIDAR',
    #             load_dim=4,
    #             use_dim=4),
    #         dict(
    #             type='LoadAnnotations3D',
    #             with_bbox_3d=False,
    #             with_label_3d=False,
    #             with_seg_3d=True,
    #             seg_3d_dtype='np.int32',
    #             seg_offset=2**16,
    #             dataset_type='semantickitti'),
    #         dict(type='PointSegClassMapping')
    #     ],
    #     prob=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(
        type='RandomJitterPoints',
        jitter_std=[0.1, 0.1, 0.1],
        clip_range=[-0.1, 0.1]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointSample', num_points=0.9, replace=True),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    batch_size=12, sampler=dict(seed=0), dataset=dict(pipeline=train_pipeline))

vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

lr = 0.24 * 2
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD', lr=lr, weight_decay=0.0001, momentum=0.9, nesterov=True),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        begin=1,
        end=36,
        by_epoch=True,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (12 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=24)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)
