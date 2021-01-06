# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.08, 0.08, 0.1]
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[-76.8, -51.2, -2, 76.8, 51.2, 4],
        voxel_size=voxel_size,
        max_voxels=(80000, 90000)),
    voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[61, 1280, 1920],
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=384,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-76.8, -51.2, -0.0345, 76.8, 51.2, -0.0345],
                    [-76.8, -51.2, 0, 76.8, 51.2, 0],
                    [-76.8, -51.2, -0.1188, 76.8, 51.2, -0.1188]],
            sizes=[
                [2.08, 4.73, 1.77],  # car
                [0.84, 0.91, 1.74],  # pedestrian
                [0.84, 1.81, 1.77]  # cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))
# model training and testing settings
train_cfg = dict(
    assigner=[
        dict(  # car
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.55,
            neg_iou_thr=0.4,
            min_pos_iou=0.4,
            ignore_iof_thr=-1),
        dict(  # pedestrian
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        dict(  # cyclist
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1)
    ],
    allowed_border=0,
    code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_pre=4096,
    nms_thr=0.25,
    score_thr=0.1,
    min_bbox_size=0,
    max_num=500)
