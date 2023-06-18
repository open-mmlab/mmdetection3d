voxel_size = [0.2, 0.2, 8]

model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel (16 in pcdet)
        point_cloud_range=[-75.2, -75.2, -5.0, 75.2, 75.2, 3.0],
        voxel_size=voxel_size,
        max_voxels=(60000, 60000)  # (training, testing) max_voxels
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]),
    middle_encoder=dict(
        # output_shape (y, x) = point_cloud_range // voxel_size (y, x)
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[752, 752]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        # OnceDataset has 5 classes
        num_classes=5,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            # ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']
            ranges=[
                [-75.2, -75.2, -1.71, 75.2, 75.2, -1.71],
                [-75.2, -75.2, -1.74, 75.2, 75.2, -1.74],
                [-75.2, -75.2, -1.55, 75.2, 75.2, -1.55],
                [-75.2, -75.2, -1.62, 75.2, 75.2, -1.62],
                [-75.2, -75.2, -1.65, 75.2, 75.2, -1.65],
            ],
            sizes=[[4.38, 1.87, 1.59], [11.11, 2.88, 3.41], [7.52, 2.50, 2.62],
                   [0.75, 0.76, 1.69], [2.18, 0.79, 1.43]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
            dict(  # for Bus
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            dict(  # for Truck
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.3,
                neg_iou_thr=0.15,
                min_pos_iou=0.15,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=4096,
        max_num=500))
