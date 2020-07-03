voxel_size = [0.16, 0.16, 4]
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
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
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -1.78, 70.4, 39.68, -1.78],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
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
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))
# model training and testing settings
train_cfg = dict(
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
    debug=False)
test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_thr=0.01,
    score_thr=0.1,
    min_bbox_size=0,
    nms_pre=100,
    max_num=50)
