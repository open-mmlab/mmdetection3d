model = dict(
    type='PointRCNN',
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, 256, 64),
        radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                    128)),
                     ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                          (256, 384, 512))),
        fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        aggregation_channels=(None, None, None, None),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    neck=dict(
        type='PointNetFPNeck',
        fp_channels=((1536, 512, 512), (768, 512, 512), (608, 256, 256),
                     (257, 128, 128))),
    rpn_head=dict(
        type='PointRPNHead',
        num_classes=3,
        enlarge_width=0.1,
        pred_layer_cfg=dict(
            in_channels=128,
            cls_linear_channels=(256, 256),
            reg_linear_channels=(256, 256)),
        cls_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        bbox_loss=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=1.0),
        bbox_coder=dict(
            type='PointXYZWHLRBBoxCoder',
            code_size=8,
            # code_size: (center residual (3), size regression (3),
            #             torch.cos(yaw) (1), torch.sin(yaw) (1)
            use_mean_size=True,
            mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6,
                                                            1.73]])),
    roi_head=dict(
        type='PointRCNNRoIHead',
        point_roi_extractor=dict(
            type='Single3DRoIPointExtractor',
            roi_layer=dict(type='RoIPointPool3d', num_sampled_points=512)),
        bbox_head=dict(
            type='PointRCNNBboxHead',
            num_classes=1,
            pred_layer_cfg=dict(
                in_channels=512,
                cls_conv_channels=(256, 256),
                reg_conv_channels=(256, 256),
                bias=True),
            in_channels=5,
            # 5 = 3 (xyz) + scores + depth
            mlp_channels=[128, 128],
            num_points=(128, 32, -1),
            radius=(0.2, 0.4, 100),
            num_samples=(16, 16, 16),
            sa_channels=((128, 128, 128), (128, 128, 256), (256, 256, 512)),
            with_corner_loss=True),
        depth_normalizer=70.0),
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=10.0,
        rpn=dict(
            nms_cfg=dict(
                use_rotate_nms=True, iou_thr=0.8, nms_pre=9000, nms_post=512),
            score_thr=None),
        rcnn=dict(
            assigner=[
                dict(  # for Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Cyclist
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.7,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_cfg=dict(
                use_rotate_nms=True, iou_thr=0.85, nms_pre=9000, nms_post=512),
            score_thr=None),
        rcnn=dict(use_rotate_nms=True, nms_thr=0.1, score_thr=0.1)))
