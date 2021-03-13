dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    type='ImVoteNet',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    img_rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    img_roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    pts_backbone=dict(
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
            normalize_xyz=True)),
    pts_bbox_heads=dict(
        common=dict(
            type='VoteHead',
            num_classes=10,
            bbox_coder=dict(
                type='PartialBinBasedBBoxCoder',
                num_sizes=10,
                num_dir_bins=12,
                with_rot=True,
                mean_sizes=[[2.114256, 1.620300, 0.927272],
                            [0.791118, 1.279516, 0.718182],
                            [0.923508, 1.867419, 0.845495],
                            [0.591958, 0.552978, 0.827272],
                            [0.699104, 0.454178, 0.75625],
                            [0.69519, 1.346299, 0.736364],
                            [0.528526, 1.002642, 1.172878],
                            [0.500618, 0.632163, 0.683424],
                            [0.404671, 1.071108, 1.688889],
                            [0.76584, 1.398258, 0.472728]]),
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
                type='SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0),
            semantic_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
        joint=dict(
            vote_module_cfg=dict(
                in_channels=512,
                vote_per_seed=1,
                gt_per_seed=3,
                conv_channels=(512, 256),
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
                mlp_channels=[512, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True)),
        pts=dict(
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
                normalize_xyz=True)),
        img=dict(
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
                normalize_xyz=True)),
        loss_weights=[0.4, 0.3, 0.3]),
    img_mlp=dict(
        in_channel=18,
        conv_channels=(256, 256),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    fusion_layer=dict(
        type='VoteFusion',
        num_classes=len(class_names),
        max_imvote_per_pixel=3),
    num_sampled_seed=1024,

    # model training and testing settings
    train_cfg=dict(
        img_rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        img_rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        img_rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        pts=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote')),
    test_cfg=dict(
        img_rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        img_rcnn=dict(
            score_thr=0.1,  # 0.05,
            nms=dict(type='nms', iou_threshold=0.5),  # 0.5),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
        pts=dict(
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.05,
            per_class_proposal=True)))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    # dict(type='LoadImageRawFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1333, 600),
        # multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='RandomFlip3D',
        sync_2d=False,  # False
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        # rot_range=[-0.0, 0.0],
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='IndoorPointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'calib'  # , 'bbox_2d'
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadImageRawFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='IndoorPointSample', num_points=20000),
            # dict(type='ImageToTensor', keys=['img']),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'img',
                    'points',
                    'calib'  # , 'bbox_2d'
                ])
        ]),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(
    #             type='RandomFlip3D',
    #             sync_2d=False,
    #             flip_ratio_bev_horizontal=0.5,
    #         ),
    #         dict(type='IndoorPointSample', num_points=20000),
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points'])
    #     ])
]

data = dict(
    samples_per_gpu=16,  # 16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train_bbox.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=False,
            # use_bbox=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_bbox.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        # use_bbox=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_bbox.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        # use_bbox=True,
        box_type_3d='Depth'))

lr = 0.008  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32])
# runtime settings
total_epochs = 36

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/imvotenet_imgpretrain_1231/latest.pth'
# 'work_dirs/imvotenet3/latest.pth'

resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
