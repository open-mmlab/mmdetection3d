_base_ = [
    '../_base_/datasets/waymoD5-mv-mono3d-3class.py',
    '../_base_/models/pgd.py', '../_base_/schedules/mmdet-schedule-1x.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(num_outs=3),
    bbox_head=dict(
        num_classes=3,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        pred_bbox2d=True,
        use_onlyreg_proj=True,
        strides=(8, 16, 32),
        regress_ranges=((-1, 128), (128, 256), (256, 1e8)),
        group_reg_dims=(2, 1, 3, 1, 16,
                        4),  # offset, depth, size, rot, kpts, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, )  # bbox2d
        ),
        centerness_branch=(256, ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        use_depth_classifier=True,
        depth_branch=(256, ),
        depth_range=(0, 50),
        depth_unit=10,
        division='uniform',
        depth_bins=6,
        pred_keypoints=True,
        weight_dim=1,
        loss_depth=dict(
            type='UncertainSmoothL1Loss', alpha=1.0, beta=3.0,
            loss_weight=1.0),
        loss_bbox2d=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.0),
        loss_consistency=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        bbox_coder=dict(
            type='PGDBBoxCoder',
            base_depths=((41.01, 18.44), ),
            base_dims=(
                (4.73, 1.77, 2.08),
                (0.91, 1.74, 0.84),
                (1.81, 1.77, 0.84),
            ),
            code_size=7)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.2 for 16-dim keypoint offsets and 1.0 for 4-dim 2D distance targets
    train_cfg=dict(code_weight=[
        1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]),
    test_cfg=dict(nms_pre=100, nms_thr=0.05, score_thr=0.001, max_per_img=20))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.008,
    ),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
total_epochs = 24
runner = dict(max_epochs=total_epochs)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
