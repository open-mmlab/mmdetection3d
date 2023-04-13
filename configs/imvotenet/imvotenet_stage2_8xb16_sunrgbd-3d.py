_base_ = [
    '../_base_/datasets/sunrgbd-3d.py', '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py', '../_base_/models/imvotenet.py'
]

class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
backend_args = None

model = dict(
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
            objectness_loss=dict(
                type='mmdet.CrossEntropyLoss',
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
                type='mmdet.CrossEntropyLoss',
                reduction='sum',
                loss_weight=1.0),
            dir_res_loss=dict(
                type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0),
            size_class_loss=dict(
                type='mmdet.CrossEntropyLoss',
                reduction='sum',
                loss_weight=1.0),
            size_res_loss=dict(
                type='mmdet.SmoothL1Loss',
                reduction='sum',
                loss_weight=10.0 / 3.0),
            semantic_loss=dict(
                type='mmdet.CrossEntropyLoss',
                reduction='sum',
                loss_weight=1.0)),
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
    freeze_img_branch=True,

    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mode='vote')),
    test_cfg=dict(
        img_rcnn=dict(score_thr=0.1),
        pts=dict(
            sample_mode='seed',
            nms_thr=0.25,
            score_thr=0.05,
            per_class_proposal=True)))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_bbox_3d=True,
        with_label_3d=True),
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(
        type='Pack3DDetInputs',
        keys=([
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'points', 'gt_bboxes_3d',
            'gt_labels_3d'
        ]))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(type='PointSample', num_points=20000),
    dict(type='Pack3DDetInputs', keys=['img', 'points'])
]

train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# may also use your own pre-trained image branch
load_from = 'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth'  # noqa
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
randomness = dict(seed=8)
