primitive_z_cfg = dict(
    type='PrimitiveHead',
    num_dims=2,
    num_classes=18,
    primitive_mode='z',
    upper_thresh=100.0,
    surface_thresh=0.5,
    vote_module_cfg=dict(
        in_channels=256,
        vote_per_seed=1,
        gt_per_seed=1,
        conv_channels=(256, 256),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        norm_feats=True,
        vote_loss=dict(
            type='ChamferDistance',
            mode='l1',
            reduction='none',
            loss_dst_weight=10.0)),
    vote_aggregation_cfg=dict(
        type='PointSAModule',
        num_point=1024,
        radius=0.3,
        num_sample=16,
        mlp_channels=[256, 128, 128, 128],
        use_xyz=True,
        normalize_xyz=True),
    feat_channels=(128, 128),
    conv_cfg=dict(type='Conv1d'),
    norm_cfg=dict(type='BN1d'),
    objectness_loss=dict(
        type='CrossEntropyLoss',
        class_weight=[0.4, 0.6],
        reduction='mean',
        loss_weight=30.0),
    center_loss=dict(
        type='ChamferDistance',
        mode='l1',
        reduction='sum',
        loss_src_weight=0.5,
        loss_dst_weight=0.5),
    semantic_reg_loss=dict(
        type='ChamferDistance',
        mode='l1',
        reduction='sum',
        loss_src_weight=0.5,
        loss_dst_weight=0.5),
    semantic_cls_loss=dict(
        type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
    train_cfg=dict(
        dist_thresh=0.2,
        var_thresh=1e-2,
        lower_thresh=1e-6,
        num_point=100,
        num_point_line=10,
        line_thresh=0.2))

primitive_xy_cfg = dict(
    type='PrimitiveHead',
    num_dims=1,
    num_classes=18,
    primitive_mode='xy',
    upper_thresh=100.0,
    surface_thresh=0.5,
    vote_module_cfg=dict(
        in_channels=256,
        vote_per_seed=1,
        gt_per_seed=1,
        conv_channels=(256, 256),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        norm_feats=True,
        vote_loss=dict(
            type='ChamferDistance',
            mode='l1',
            reduction='none',
            loss_dst_weight=10.0)),
    vote_aggregation_cfg=dict(
        type='PointSAModule',
        num_point=1024,
        radius=0.3,
        num_sample=16,
        mlp_channels=[256, 128, 128, 128],
        use_xyz=True,
        normalize_xyz=True),
    feat_channels=(128, 128),
    conv_cfg=dict(type='Conv1d'),
    norm_cfg=dict(type='BN1d'),
    objectness_loss=dict(
        type='CrossEntropyLoss',
        class_weight=[0.4, 0.6],
        reduction='mean',
        loss_weight=30.0),
    center_loss=dict(
        type='ChamferDistance',
        mode='l1',
        reduction='sum',
        loss_src_weight=0.5,
        loss_dst_weight=0.5),
    semantic_reg_loss=dict(
        type='ChamferDistance',
        mode='l1',
        reduction='sum',
        loss_src_weight=0.5,
        loss_dst_weight=0.5),
    semantic_cls_loss=dict(
        type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
    train_cfg=dict(
        dist_thresh=0.2,
        var_thresh=1e-2,
        lower_thresh=1e-6,
        num_point=100,
        num_point_line=10,
        line_thresh=0.2))

primitive_line_cfg = dict(
    type='PrimitiveHead',
    num_dims=0,
    num_classes=18,
    primitive_mode='line',
    upper_thresh=100.0,
    surface_thresh=0.5,
    vote_module_cfg=dict(
        in_channels=256,
        vote_per_seed=1,
        gt_per_seed=1,
        conv_channels=(256, 256),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        norm_feats=True,
        vote_loss=dict(
            type='ChamferDistance',
            mode='l1',
            reduction='none',
            loss_dst_weight=10.0)),
    vote_aggregation_cfg=dict(
        type='PointSAModule',
        num_point=1024,
        radius=0.3,
        num_sample=16,
        mlp_channels=[256, 128, 128, 128],
        use_xyz=True,
        normalize_xyz=True),
    feat_channels=(128, 128),
    conv_cfg=dict(type='Conv1d'),
    norm_cfg=dict(type='BN1d'),
    objectness_loss=dict(
        type='CrossEntropyLoss',
        class_weight=[0.4, 0.6],
        reduction='mean',
        loss_weight=30.0),
    center_loss=dict(
        type='ChamferDistance',
        mode='l1',
        reduction='sum',
        loss_src_weight=1.0,
        loss_dst_weight=1.0),
    semantic_reg_loss=dict(
        type='ChamferDistance',
        mode='l1',
        reduction='sum',
        loss_src_weight=1.0,
        loss_dst_weight=1.0),
    semantic_cls_loss=dict(
        type='CrossEntropyLoss', reduction='sum', loss_weight=2.0),
    train_cfg=dict(
        dist_thresh=0.2,
        var_thresh=1e-2,
        lower_thresh=1e-6,
        num_point=100,
        num_point_line=10,
        line_thresh=0.2))

model = dict(
    type='H3DNet',
    backbone=dict(
        type='MultiBackbone',
        num_streams=4,
        suffixes=['net0', 'net1', 'net2', 'net3'],
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        backbones=dict(
            type='PointNet2SASSG',
            in_channels=4,
            num_points=(2048, 1024, 512, 256),
            radius=(0.2, 0.4, 0.8, 1.2),
            num_samples=(64, 32, 16, 16),
            sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                         (128, 128, 256)),
            fp_channels=((256, 256), (256, 256)),
            norm_cfg=dict(type='BN2d'),
            sa_cfg=dict(
                type='PointSAModule',
                pool_mod='max',
                use_xyz=True,
                normalize_xyz=True))),
    rpn_head=dict(
        type='VoteHead',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    roi_head=dict(
        type='H3DRoIHead',
        primitive_list=[primitive_z_cfg, primitive_xy_cfg, primitive_line_cfg],
        bbox_head=dict(
            type='H3DBboxHead',
            gt_per_seed=3,
            num_proposal=256,
            suface_matching_cfg=dict(
                type='PointSAModule',
                num_point=256 * 6,
                radius=0.5,
                num_sample=32,
                mlp_channels=[128 + 6, 128, 64, 32],
                use_xyz=True,
                normalize_xyz=True),
            line_matching_cfg=dict(
                type='PointSAModule',
                num_point=256 * 12,
                radius=0.5,
                num_sample=32,
                mlp_channels=[128 + 12, 128, 64, 32],
                use_xyz=True,
                normalize_xyz=True),
            feat_channels=(128, 128),
            primitive_refine_channels=[128, 128, 128],
            upper_thresh=100.0,
            surface_thresh=0.5,
            line_thresh=0.5,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            objectness_loss=dict(
                type='CrossEntropyLoss',
                class_weight=[0.2, 0.8],
                reduction='sum',
                loss_weight=5.0),
            center_loss=dict(
                type='ChamferDistance',
                mode='l2',
                reduction='sum',
                loss_src_weight=10.0,
                loss_dst_weight=10.0),
            dir_class_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=0.1),
            dir_res_loss=dict(
                type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
            size_class_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=0.1),
            size_res_loss=dict(
                type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
            semantic_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=0.1),
            cues_objectness_loss=dict(
                type='CrossEntropyLoss',
                class_weight=[0.3, 0.7],
                reduction='mean',
                loss_weight=5.0),
            cues_semantic_loss=dict(
                type='CrossEntropyLoss',
                class_weight=[0.3, 0.7],
                reduction='mean',
                loss_weight=5.0),
            proposal_objectness_loss=dict(
                type='CrossEntropyLoss',
                class_weight=[0.2, 0.8],
                reduction='none',
                loss_weight=5.0),
            primitive_center_loss=dict(
                type='MSELoss', reduction='none', loss_weight=1.0))))

# model training and testing settings
train_cfg = dict(
    rpn=dict(pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote'),
    rpn_proposal=dict(use_nms=False),
    rcnn=dict(
        pos_distance_thr=0.3,
        neg_distance_thr=0.6,
        sample_mod='vote',
        far_threshold=0.6,
        near_threshold=0.3,
        mask_surface_threshold=0.3,
        label_surface_threshold=0.3,
        mask_line_threshold=0.3,
        label_line_threshold=0.3))

test_cfg = dict(
    rpn=dict(
        sample_mod='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True,
        use_nms=False),
    rcnn=dict(
        sample_mod='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True))
