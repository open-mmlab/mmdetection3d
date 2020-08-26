model = dict(
    type='SSD3DNet',
    backbone=dict(
        type='SSDSAMSG',
        in_channels=4,
        num_points=(4096, 512, (256, 256)),
        radius=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 192, 256), (128, 256, 256))),
        aggregation_channels=(64, 128, 256),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (512, -1)),
        norm_cfg=dict(type='BN2d'),
        pool_mod='max'),
    bbox_head=dict(
        type='SSD3DHead',
        in_channels=256,
        num_candidates=256,
        vote_aggregation_cfg=dict(
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 512), (256, 512, 1024)),
            use_xyz=True,
            normalize_xyz=True),
        feat_channels=(128, ),
        cls_pred_mlps=(128, ),
        reg_pred_mlps=(128, ),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=2.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=2.0),
        corner_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(pos_distance_thr=10.0, expand_dims_length=0.1)
test_cfg = dict(
    nms_thr=0.1,
    score_thr=0.05,
    per_class_proposal=True,
    max_translate_range=(3.0, 2.0, 3.0))
