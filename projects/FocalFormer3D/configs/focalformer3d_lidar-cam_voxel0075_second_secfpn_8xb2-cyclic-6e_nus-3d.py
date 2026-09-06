# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# FocalFormer3D, LiDAR + camera, nuScenes.
# Ported from NVlabs/FocalFormer3D,
# projects/configs/focalformer3d/FocalFormer3D_LC.py
# to the mmdet3d 1.x (mmengine) config format.
#
# Self-contained on purpose: it does not inherit from the LiDAR-only config.
# The camera branch changes the fusion variant, the number of fusion blocks and
# the head's heatmap stages, so an inheriting version was mostly overrides and
# it was easy to misread which value was in force.
#
# Where the camera-specific values come from
# ------------------------------------------
# There is no camera block in the LiDAR-only config, so every value below was
# either taken from upstream or read off the released LC checkpoint:
#
#   img_backbone  ResNet-50   conv1 is (64, 3, 7, 7); layer1..4 are 3/4/6/3
#                             bottleneck blocks, so R50, not R18/R101.
#   img_neck      FPN         lateral_convs in_channels are 256/512/1024/2048,
#                             all out 256. Only 4 fpn_convs carry weights,
#                             consistent with num_outs=5: the fifth level is a
#                             weightless max-pool.
#   img_scale     448 x 800   upstream writes this as (800, 448) and transposes
#                             it at the call site; it is (H, W) here because
#                             that is what LiftSplatShoot expects.
#
# The released checkpoint ships in upstream key naming
# (pts_bbox_head.decoder.*.attentions.0.attn.*). Convert it with
# tools/convert_focalformer_ckpt.py before loading.
#
# How the camera reaches the BEV
# ------------------------------
# FocalEncoder.forward does not consume images directly. With cam_lss truthy
# (and not the literal string 'proj') it builds per-camera rotations and
# translations by INVERTING img_metas['lidar2img'], then hands those to
# LiftSplatShoot. LiftSplatShoot.get_geometry separately reads 'img_aug_matrix'
# to undo the resize and crop this pipeline applies. Both keys must survive
# into metainfo: dropping either one leaves camera features splatted to the
# wrong BEV cells, which costs mAP without raising anything.
#
# Expected clean eval: mAP 0.705 / NDS 0.731, the checkpoint's own reference.
# The LiDAR-only config reproduces its reference to within 0.6 of a point, so
# treat a gap larger than about 1 point here as a wiring failure.
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
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# (H, W), NOT (W, H). Upstream writes (800, 448) and transposes at the call
# site. LiftSplatShoot is height-first (its own default is (900, 1600), the
# nuScenes H x W). Passing (800, 448) here builds a frustum of
# (41, 200, 112, 3) against the checkpoint's (41, 112, 200, 3), i.e. the image
# plane transposed. It loads with only a shape warning and then splats every
# camera feature to the wrong BEV cell.
img_scale = (448, 800)
final_dim = [448, 800]
# nuScenes images are 1600x900. 0.5 gives 800x450; FFImageAug3D then crops 2
# rows off the top to reach 448, which is why bot_pct_lim is [0, 0].
test_resize_lim = [0.5, 0.5]
# Upstream's training-time jitter. Preserves aspect ratio; the crop is allowed
# to run off the top of the resized image, and PIL pads those rows with black.
train_resize_lim = [0.4, 0.6]
num_views = 6

multistage_heatmap = 2
inter_channel = 128
extra_feat = True

# Both matrices are read by LiftSplatShoot.get_geometry; see the header.
meta_keys = [
    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
    'lidar_path', 'img_path', 'num_pts_feats'
]

# Images are loaded AFTER the point-cloud augmentations on purpose.
# RandomFlip3D calls mmdet's RandomFlip whenever 'img' is already in the
# results dict, which would flip the views without updating lidar2img and
# desynchronise the camera-to-BEV projection. Loading afterwards keeps
# FFImageAug3D the only transform that touches the images, and it records
# everything it does in img_aug_matrix so LSS can undo it.
#
# Known limitation, inherited from upstream: GlobalRotScaleTrans and
# RandomFlip3D move the points but nothing recomputes lidar2img for them, and
# this LiftSplatShoot does not read a lidar_aug_matrix the way
# projects/BEVFusion does. Upstream trains this stage with the LiDAR branch,
# the image branch and the LSS all frozen, so only the decoder sees the
# mismatch. Freeze them here too, or drop the two transforms.
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
        type='FFLoadMultiViewImage',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='FFImageAug3D',
        final_dim=final_dim,
        resize_lim=train_resize_lim,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=meta_keys),
]

# Deliberately NOT MultiScaleFlipAug3D, which the LiDAR-only test pipeline
# uses: it wraps the point transforms in a nested structure that carries no
# image branch, so the camera metadata would never reach the model. This
# mirrors the BEVFusion lidar-cam pipeline, the known-good camera path in this
# repository.
test_pipeline = [
    dict(
        type='FFLoadMultiViewImage',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
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
        type='FFImageAug3D',
        final_dim=final_dim,
        resize_lim=test_resize_lim,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'points'], meta_keys=meta_keys),
]

# data_prefix must name every camera, or FFLoadMultiViewImage cannot resolve
# img_path and the run dies on the first sample.
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT')

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
            data_prefix=data_prefix,
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
        data_prefix=data_prefix,
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
# preprocessor runs with voxel=False and only normalises, collates and pads.
model = dict(
    type='FocalFormer3D',
    # Upstream trains this stage with everything but the decoder frozen.
    freeze_img=True,
    freeze_camlss=True,
    freeze_pts=True,
    input_img=True,
    # bgr_to_rgb=False deliberately. FFLoadMultiViewImage decodes with
    # channel_order='rgb', so the images are ALREADY RGB here. Upstream's
    # img_norm_cfg to_rgb=True describes the mmdet 2.x pipeline doing that
    # conversion; repeating it would feed BGR to a model trained on RGB.
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        voxel=False,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
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
        # Two fusion blocks: the checkpoint carries fusion_blocks.0 AND
        # fusion_blocks.1. The LiDAR-only config runs one.
        num_layers=multistage_heatmap,
        in_channels_img=256,
        in_channels_pts=sum([256, 256]),
        hidden_channel=inter_channel,
        bn_momentum=0.1,
        max_points_height=10,
        bias='auto',
        # 'bevfusion', not the LiDAR-only config's 'bevfusionmb2'. The two
        # branch in FocalEncoderLayer.__init__: 'bevfusionmb2' builds
        # P_IML/P_integration as MobileNetV2 InvertedResiduals,
        # 'bevfusion' builds them as LocalContextAttentionBlock + ConvBNReLU,
        # which is what carries key_project/query_project. The LC checkpoint
        # has the latter, so inheriting the LiDAR-only value would silently
        # mismatch 146 tensors.
        iterbev='bevfusion',
        cam_lss=True,
        # Required whenever cam_lss is on, and it costs no parameters.
        # FocalEncoder passes need_projbev=not cam_lss, so with LSS the I2P
        # block is deliberately not built: the camera features are already in
        # BEV and there is nothing to project. The checkpoint agrees, it has
        # no I2P_block weights. But FocalEncoderLayer.forward only honours
        # that in its iter_bev_cam branch; the other path calls
        # self.I2P_block(...) unguarded and dies on 'NoneType' object is not
        # callable. With iter_bev_cam=True both layers take
        # I2P_feat = img_feat, i.e. the LSS BEV output directly.
        iter_bev_cam=True,
        pc_range=point_cloud_range,
        img_scale=img_scale,
        multistage_heatmap=multistage_heatmap,
        extra_feat=extra_feat),
    pts_bbox_head=dict(
        type='FocalDecoder',
        # The checkpoint carries heatmap_head_img.0.* AND .1.*, both real.
        # FocalDecoder.__init__ does `if reuse_first_heatmap:
        # multistage_heatmap += 1` and then appends None at index 0, so the
        # LiDAR-only settings (1, reuse=True) build [None, copy], with weights
        # only at .1. Two real stages means this head was built with
        # multistage_heatmap=2 and reuse off.
        multistage_heatmap=multistage_heatmap,
        reuse_first_heatmap=False,
        extra_feat=extra_feat,
        roi_feats=7,
        roi_dropout_rate=0.1,
        roi_based_reg=True,
        roi_expand_ratio=1.2,
        heatmap_box=False,
        thin_heatmap_box=False,
        multiscale=True,
        mask_heatmap_mode='poscls',
        input_img=True,
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

# Upstream runs this stage from two pretrained checkpoints: the LiDAR branch
# from DeformFormer3D_L (the deformformer3d config in this directory) and
# the image branch from a
# camera-only DeformFormer3D_C_R50. mmengine has a single load_from, so merge
# the two into one checkpoint first, or point this at the LiDAR one and let
# img_backbone start from its ImageNet init.
load_from = None
resume = False

find_unused_parameters = True

# No custom_hooks. Upstream's LC train pipeline has no ObjectSample, so there
# is nothing for Fading to remove.

auto_scale_lr = dict(enable=False, base_batch_size=16)
