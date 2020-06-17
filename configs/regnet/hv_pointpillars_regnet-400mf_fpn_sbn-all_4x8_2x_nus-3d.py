_base_ = [
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# model settings
voxel_size = [0.25, 0.25, 8]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
model = dict(
    type='MVXFasterRCNNV2',
    pretrained=dict(pts='open-mmlab://regnetx_400mf'),
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
    pts_backbone=dict(
        type='NoStemRegNet',
        arch=dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, bot_mul=1.0),
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        strides=(1, 2, 2, 2),
        base_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 160, 384],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
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
    pts=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    pts=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))

# dataset settings
input_modality = dict(
    use_lidar=True,
    use_depth=False,
    use_lidar_intensity=True,
    use_camera=False)

data = dict(
    train=dict(modality=input_modality),
    val=dict(modality=input_modality),
    test=dict(modality=input_modality))

evaluation = dict(interval=24)
