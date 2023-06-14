_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.DSVT.dsvt'], allow_failed_imports=False)

voxel_size = [0.32, 0.32, 6]
grid_size = [468, 468, 1]
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4.0]
data_root = 'data/waymo/kitti_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

model = dict(
    type='DSVT',
    data_preprocessor=dict(type='Det3DDataPreprocessor', voxel=False),
    voxel_encoder=dict(
        type='DynamicPillarVFE3D',
        with_distance=False,
        use_absolute_xyz=True,
        use_norm=True,
        num_filters=[192, 192],
        num_point_features=5,
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='DSVTMiddleEncoder',
        input_layer=dict(
            sparse_shape=grid_size,
            downsample_stride=[],
            dim_model=[192],
            set_info=[[36, 4]],
            window_shape=[[12, 12, 1]],
            hybrid_factor=[2, 2, 1],  # x, y, z
            shift_list=[[[0, 0, 0], [6, 6, 0]]],
            normalize_pos=False),
        set_info=[[36, 4]],
        dim_model=[192],
        dim_feedforward=[384],
        stage_num=1,
        nhead=[8],
        conv_out_channel=192,
        output_shape=[468, 468],
        dropout=0.,
        activation='gelu'),
    map2bev=dict(
        type='PointPillarsScatter3D',
        output_shape=grid_size,
        num_bev_feats=192),
    backbone=dict(
        type='ResSECOND',
        in_channels=192,
        out_channels=[128, 128, 256],
        blocks_nums=[1, 2, 2],
        layer_strides=[1, 2, 2]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=False),
    bbox_head=dict(
        type='DSVTCenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[dict(num_class=3, class_names=class_names)],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), iou=(1, 2)),
        share_conv_channel=64,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        bbox_coder=dict(
            type='DSVTBBoxCoder',
            pc_range=point_cloud_range,
            max_num=500,
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01)),
        loss_cls=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=2.0),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        iou_rectifier=[[0.68, 0.71, 0.65]],
        pc_range=[-80, -80],
        out_size_factor=4,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        multi_class_nms=True,
        pre_max_size=[[4096, 4096, 4096]],
        post_max_size=[[500, 500, 500]],
        nms_thr=[[0.7, 0.6, 0.55]]))

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
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        norm_intensity=True,
        backend_args=backend_args),
    # Add this if using `MultiFrameDeformableDecoderRPN`
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
        norm_elongation=True,
        backend_args=backend_args),
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
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
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
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/gt.bin',
    data_root='./data/waymo/waymo_format',
    backend_args=backend_args,
    convert_kitti_format=False,
    idx2metainfo='./data/waymo/waymo_format/idx2metainfo.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime settings
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (1 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=8)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
