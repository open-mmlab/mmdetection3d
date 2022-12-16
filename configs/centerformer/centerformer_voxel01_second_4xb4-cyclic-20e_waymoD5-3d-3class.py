_base_ = ['../_base_/default_runtime.py']

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.1, 0.1, 0.15]
point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')

model = dict(
    type='CenterFormer',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1))),
    voxel_encoder=dict(
        type='DynamicSimpleVFE',
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1504, 1504],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, [0, 1, 1]), (1, 1)),
        block_type='basicblock'),
    backbone=dict(
        type='SECOND_WithAtten',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[256, 256],
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
    # neck return up, down, norm 3 stirdes,
    neck=dict(
        type='SECONDFPN_WithAtten',
        in_channels=[256, 256],
        upsample_strides=[2, 4],
        out_channels=[128, 128],
        upsample_cfg=dict(type='deconv', bias=False),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        return_inputs=True),
    bbox_head=dict(
        type='CenterFormerHead',
        in_channels=256,
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'cyclist'])
        ],
        transformer_config=dict(
            depth=2,
            heads=6,
            dim_head=64,
            MLP_dim=256,
            DP_rate=0.3,
            out_att=False,
            n_points=15,
        ),
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), iou=(1, 2)),
        num_heatmap_convs=2,
        num_cornermap_convs=2,
        share_conv_channel=64,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=1,
            # TODO: set False
            bias=True,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='naiveSyncBN1d')),
        loss_cls=dict(type='FastFocalLoss'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=2),
        loss_corner=dict(
            type='mmdet.MSELoss', reduction='mean', loss_weight=1),
        loss_iou=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=1)),
    train_cfg=dict(
        grid_size=[1504, 1504, 40],
        voxel_size=voxel_size,
        out_size_factor=4,
        gaussian_overlap=0.1,
        point_cloud_range=point_cloud_range,
        num_center_proposals=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    test_cfg=dict(
        post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
        point_cloud_range=point_cloud_range,
        num_center_proposals=1000,
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=voxel_size[:2],
        iou_factor=[1, 1, 4],
        nms_type='rotate',
        use_multi_class_nms=True,
        nms_pre_max_size=[1600, 1600, 800],
        nms_post_max_size=[200, 200, 100],
        nms_iou_thres=[0.8, 0.55, 0.55],
    ))

data_root = 'data/waymo_mini/kitti_format/'
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4]))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        norm_intensity=True),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     load_dim=6,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.5, 0.5, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        norm_intensity=True,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

dataset_type = 'WaymoDataset'
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        # load one frame every five frames
        load_interval=5,
        file_client_args=file_client_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/gt.bin',
    data_root='./data/waymo/waymo_format',
    file_client_args=file_client_args,
    convert_kitti_format=False,
    idx2metainfo='./data/waymo/waymo_format/idx2metainfo.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# For waymo dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 3e-4
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01, betas=(0.9, 0.99)),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=20)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]

load_from = './checkpoints/centerformer_our_refactor.pth'
