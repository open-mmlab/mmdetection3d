_base_ = ['./minkunet_w32_8xb2-15e_semantickitti.py']

model = dict(
    backbone=dict(type='MinkUNetBackboneV2', encoder_blocks=[2, 3, 4, 6]),
    decode_head=dict(channels=256 + 128 + 96))

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
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='LaserMix',
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-25, 3],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
            [
                dict(
                    type='PolarMix',
                    instance_classes=[1, 2, 3, 4, 5, 6, 7, 8],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
        ],
        prob=[0.5, 0.5]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.008
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=10, norm_type=2))

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.1)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
