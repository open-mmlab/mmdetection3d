dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle']
point_cloud_range = [0, -55.2, -15, 128, 55.2, 15]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(
            Car=5, Pedestrian=10, Cyclist=10, Large_vehicle=10)),
    classes=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10, Large_vehicle=10))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='data/kitti/',
            info_path='data/kitti/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_min_points=dict(
                    Car=5, Pedestrian=10, Cyclist=10, Large_vehicle=10)),
            classes=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
            sample_groups=dict(
                Car=15, Pedestrian=10, Cyclist=10, Large_vehicle=10))),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='data/kitti/kitti_infos_trainval.pkl',
            split='training',
            pts_prefix='velodyne',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='data/kitti/',
                        info_path='data/kitti/kitti_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_min_points=dict(
                                Car=5,
                                Pedestrian=10,
                                Cyclist=10,
                                Large_vehicle=10)),
                        classes=[
                            'Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'
                        ],
                        sample_groups=dict(
                            Car=15,
                            Pedestrian=10,
                            Cyclist=10,
                            Large_vehicle=10))),
                dict(
                    type='ObjectNoise',
                    num_try=100,
                    translation_std=[0.25, 0.25, 0.25],
                    global_rot_range=[0.0, 0.0],
                    rot_range=[-0.15707963267, 0.15707963267]),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_test.pkl',
        split='testing',
        pts_prefix='velodyne',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_test.pkl',
        split='testing',
        pts_prefix='velodyne',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=5,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car', 'Large_vehicle'],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
voxel_size_nus = [0.2, 0.2, 8]
voxel_size_lum = [0.2, 0.2, 16]
voxel_size = [0.2, 0.2, 16]
point_cloud_range_kitti_nus = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range_kitti_lum = [0, -55.2, -8, 128, 55.2, 8]
point_cloud_range_kitti_lum_post_center = [-10, -65.2, -13, 138, 65.2, 13]
y_scatter_shape = 552
x_scatter_shape = 640
output_scatter_shape = [552, 640]
new_outsize_factor = 4
pillar_feature_in_channels_new = 4
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=[0.2, 0.2, 16],
        max_voxels=(30000, 40000),
        point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 16),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        legacy=False,
        point_cloud_range=[0, -55.2, -15, 128, 55.2, 15]),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(552, 640)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Large_vehicle'])
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-10, -65.2, -13, 138, 65.2, 13],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2],
            code_size=7,
            pc_range=[0, -55.2]),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[640, 552, 1],
            voxel_size=[0.2, 0.2, 16],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=[0, -55.2, -15, 128, 55.2, 15])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-10, -65.2, -13, 138, 65.2, 13],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[0, -55.2],
            out_size_factor=4,
            voxel_size=[0.2, 0.2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=210)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './train_0701_centerpoint_pointpillars_post80_2/'
load_from = None
resume_from = './train_0701_centerpoint_pointpillars_post80_2/latest.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
total_epochs = 210
gpu_ids = range(0, 8)
conversion = dict(
    trace_input_shapes = [(16000, 20, 4), (16000,), (16000, 4)]
)