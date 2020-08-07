_base_ = [
    '../_base_/datasets/scannet-3d-18class.py', '../_base_/models/h3dnet.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

primitive_z_cfg = dict(
    type='PrimitiveHead',
    num_dim=2,
    num_classes=18,
    primitive_mode='z',
    vote_moudule_cfg=dict(
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
        reduction='none',
        loss_src_weight=0.5,
        loss_dst_weight=0.5),
    semantic_loss=dict(
        type='CrossEntropyLoss', reduction='none', loss_weight=1.0),
    train_cfg=dict(
        dist_thresh=0.2,
        var_thresh=1e-2,
        lower_thresh=1e-6,
        num_point=100,
        num_point_line=10,
        line_thresh=0.2))

primitive_xy_cfg = dict(
    type='PrimitiveHead',
    num_dim=1,
    num_classes=18,
    primitive_mode='xy',
    vote_moudule_cfg=dict(
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
        reduction='none',
        loss_src_weight=0.5,
        loss_dst_weight=0.5),
    semantic_loss=dict(
        type='CrossEntropyLoss', reduction='none', loss_weight=1.0),
    train_cfg=dict(
        dist_thresh=0.2,
        var_thresh=1e-2,
        lower_thresh=1e-6,
        num_point=100,
        num_point_line=10,
        line_thresh=0.2))

primitive_line_cfg = dict(
    type='PrimitiveHead',
    num_dim=0,
    num_classes=18,
    primitive_mode='line',
    vote_moudule_cfg=dict(
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
        reduction='none',
        loss_src_weight=1.0,
        loss_dst_weight=1.0),
    semantic_loss=dict(
        type='CrossEntropyLoss', reduction='none', loss_weight=2.0),
    train_cfg=dict(
        dist_thresh=0.2,
        var_thresh=1e-2,
        lower_thresh=1e-6,
        num_point=100,
        num_point_line=10,
        line_thresh=0.2))

# model settings
model = dict(
    backbone=dict(
        type='MultiBackbone',
        num_stream=4,
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
            pool_mod='max',
            suffix='')),
    bbox_head=dict(
        type='H3dHead',
        num_classes=18,
        primitive_list=[primitive_z_cfg, primitive_xy_cfg, primitive_line_cfg],
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=18,
            num_dir_bins=24,
            with_rot=False,
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]])))

data = dict(samples_per_gpu=1, workers_per_gpu=1)

# optimizer
lr = 0.01  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[16, 28, 40, 48])
# runtime settings
total_epochs = 48

# optimizer
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
