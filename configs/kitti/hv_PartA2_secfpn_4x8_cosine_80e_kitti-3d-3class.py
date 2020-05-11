# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # velodyne coordinates, x, y, z

model = dict(
    type='PartA2',
    voxel_layer=dict(
        max_num_points=5,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)  # (training, testing) max_coxels
    ),
    voxel_encoder=dict(type='VoxelFeatureExtractorV3'),
    middle_encoder=dict(
        type='SparseUNet',
        in_channels=4,
        output_shape=[41, 1600, 1408],
        pre_act=False,
    ),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        num_filters=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='PartA2RPNHead',
        class_name=['Pedestrian', 'Cyclist', 'Car'],
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        encode_bg_as_zeros=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            strides=[2],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=9000,
        nms_post=512,
        nms_thr=0.8,
        score_thr=0,
        use_rotate_nms=False),
)
test_cfg = dict(
    rpn=dict(
        nms_pre=1024,
        max_per_img=100,
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.7,
        score_thr=0))

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_modality = dict(
    use_lidar=True, use_depth=False, use_lidar_intensity=True, use_camera=True)
db_sampler = dict(
    root_path=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    use_road_plane=False,
    object_rot_range=[0.0, 0.0],
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
)
train_pipeline = [
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        loc_noise_std=[0, 0, 0],
        global_rot_range=[0.0, 0.0],
        rot_uniform_noise=[-0.39269908, 0.39269908]),
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(
        type='GlobalRotScale',
        rot_uniform_noise=[-0.78539816, 0.78539816],
        scaling_uniform_noise=[0.95, 1.05],
        trans_normal_noise=[0.2, 0.2, 0.2]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file=data_root + 'kitti_infos_train.pkl',
        split='training',
        training=True,
        pipeline=train_pipeline,
        modality=input_modality,
        class_names=class_names,
        with_label=True),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        class_names=class_names,
        with_label=True),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='testing',
        pipeline=test_pipeline,
        modality=input_modality,
        class_names=class_names,
        with_label=True))
# optimizer
lr = 0.003  # max learning rate
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    target_lr=1e-5,
    as_ratio=True)
momentum_config = None
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 80
dist_params = dict(backend='nccl', port=29502)
log_level = 'INFO'
work_dir = './work_dirs/parta2_secfpn_80e'
load_from = None
resume_from = None
workflow = [('train', 1)]
