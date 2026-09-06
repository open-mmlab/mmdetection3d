# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# FocalFormer3D, LiDAR-only, nuScenes.
# Ported from NVlabs/FocalFormer3D,
# projects/configs/focalformer3d/FocalFormer3D_L.py
# to the mmdet3d 1.x (mmengine) config format.
custom_imports = dict(
    imports=['projects.FocalFormer3D.focalformer3d'],
    allow_failed_imports=False)

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

img_scale = (800, 448)
num_views = 6
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multistage_heatmap = 1
inter_channel = 128
extra_feat = True

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=dict(
            pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# FocalFormer3D voxelizes internally through pts_voxel_layer, so the
# preprocessor runs with voxel=False and only collates and pads.
model = dict(
    type='FocalFormer3D',
    freeze_img=True,
    freeze_pts=True,
    input_img=False,
    data_preprocessor=dict(type='mmdet3d.Det3DDataPreprocessor', voxel=False),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    imgpts_neck=dict(
        type='FocalEncoder',
        num_layers=multistage_heatmap,
        in_channels_img=256,
        in_channels_pts=sum([256, 256]),
        hidden_channel=inter_channel,
        bn_momentum=0.1,
        max_points_height=10,
        bias='auto',
        iterbev='bevfusionmb2',
        input_img=False,
        iterbev_wo_img=True,
        multistage_heatmap=multistage_heatmap,
        extra_feat=extra_feat),
    pts_bbox_head=dict(
        type='FocalDecoder',
        reuse_first_heatmap=True,
        extra_feat=extra_feat,
        roi_feats=7,
        roi_dropout_rate=0.1,
        roi_based_reg=True,
        roi_expand_ratio=1.2,
        heatmap_box=False,
        thin_heatmap_box=False,
        multiscale=True,
        multistage_heatmap=multistage_heatmap,
        mask_heatmap_mode='poscls',
        input_img=False,
        iterbev_wo_img=True,
        add_gt_groups=3,
        add_gt_groups_noise='box,1',
        add_gt_groups_noise_box='gtnoise',
        add_gt_pos_thresh=5.,
        add_gt_pos_boxnoise_thresh=0.75,
        gt_center_limit=5,
        bevpos=True,
        loss_weight_heatmap=1.,
        loss_weight_separate_heatmap=0.,
        loss_weight_separate_bbox=0.3,
        num_proposals=300,
        hidden_channel=inter_channel,
        num_classes=len(class_names),
        num_decoder_layers=2,
        num_heads=8,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        # decoder_cfg is built by mmcv's build_transformer_layer_sequence, not
        # by MODELS.build, so these types resolve through mmcv's
        # TRANSFORMER_LAYER_SEQUENCE / TRANSFORMER_LAYER registries.
        decoder_cfg=dict(
            type='DeformableDetrTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            post_norm_cfg=None,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=inter_channel,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=inter_channel,
                    num_levels=3,
                    num_points=4,
                    num_heads=8,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=inter_channel,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))))),
    # Model-level train_cfg / test_cfg, not the runner loop configs below.
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# Equivalent of the upstream cyclic policy:
# target_ratio=(10, 0.0001), step_ratio_up=0.4, total_epochs=6.
total_epochs = 6
up_ratio = 0.4
lr = 0.0001

param_scheduler = [
    # LR up: base_lr -> 10 * base_lr over epochs 0 -> 2.4
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs * up_ratio,
        eta_min=lr * 10,
        begin=0,
        end=total_epochs * up_ratio,
        by_epoch=True,
        convert_to_iter_based=True),
    # LR down: 10 * base_lr -> 1e-4 * base_lr over epochs 2.4 -> 6
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs * (1 - up_ratio),
        eta_min=lr * 0.0001,
        begin=total_epochs * up_ratio,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    # Momentum down: 1 -> 0.85/0.95 over epochs 0 -> 2.4
    dict(
        type='CosineAnnealingMomentum',
        T_max=total_epochs * up_ratio,
        eta_min=0.8947368421052632,
        begin=0,
        end=total_epochs * up_ratio,
        by_epoch=True,
        convert_to_iter_based=True),
    # Momentum up: 0.85/0.95 -> 1 over epochs 2.4 -> 6
    dict(
        type='CosineAnnealingMomentum',
        T_max=total_epochs * (1 - up_ratio),
        eta_min=1,
        begin=total_epochs * up_ratio,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Upstream initialises this stage from a trained DeformFormer3D_L, see
# the deformformer3d config in this directory.
load_from = None
resume = False

find_unused_parameters = True

# Fading is the plugin's own hook and matches that schedule exactly.
# DisableObjectSampleHook is the
# built-in equivalent if you prefer it.
custom_hooks = [dict(type='Fading', fade_epoch=1)]

auto_scale_lr = dict(enable=False, base_batch_size=16)
