# model settings
_base_ = [
    '../_base_/datasets/fern3d.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [ 0, -39.68, -1, 69.12, 39.68, 3]

anchors_info = {
    "truck": {
        "ranges": [0., -20.0, -1.0, 50.0, 20.0, 0.5],
        "sizes": [5.94, 2.65, 3.65],
    },
    "trailer": {
        "ranges": [0., -30.0, -1.0, 50.0, 30.0, 0.5],
        "sizes": [12.8, 2.75, 3.38],
    },
    "human": {
        "ranges": [0., -15.0, -1.0, 35.0, 15.0, 1.0],
        "sizes": [0.6, 0.6, 1.78],
    },
    "car": {
        "ranges": [0., -20.0, -1.0, 50.0, 20.0, 0.5],
        "sizes": [4.43, 1.77, 1.70],
    },
    "crane": {
        "ranges": [0., -20.0, -1.0, 50.0, 20.0, 0.5],
        "sizes": [10.5, 1.0, 2.2],
    },
    "forklift": {
        "ranges": [0., -20.0, -1.0, 40.0, 20.0, 0.5],
        "sizes": [2.5, 1.2, 1.93],
    },
    "reach_stacker": {
        "ranges": [0., -20.0, -1.0, 50.0, 20.0, 0.5],
        "sizes": [6.0, 2.5, 2.5],
    },
}

class_names = ['truck',]
# todo figure-out order of anchors vs order of classes in meta of pickle vs order of classes in config
# from KITTI it looks like order in the config is the main

# TODO define ranges
voxel_size = [0.16, 0.16, 4]
scatter_shape = [496, 432]
#point_cloud_range = [ 4, -39.68, -1, 73.12, 39.68, 3]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=scatter_shape),
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
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                anchors_info['truck']['ranges'],
            ],
            sizes=[
                anchors_info['truck']['sizes'],
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
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
        nms_pre=100,
        max_num=50))

work_dir = '/home/omuratov/bigdata/training_info/fernnet'

default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))