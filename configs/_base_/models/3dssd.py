model = dict(
    type='SSD3DNet',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 512, (256, 256)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 192, 256), (128, 256, 256))),
        aggregation_channels=(64, 128, 256),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (512, -1)),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    bbox_head=dict(
        type='SSD3DHead',
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128, ),
            reg_conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        sample_mode='spec', pos_distance_thr=10.0, expand_dims_length=0.05),
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mode='spec',
        score_thr=0.0,
        per_class_proposal=True,
        max_output_num=100))
