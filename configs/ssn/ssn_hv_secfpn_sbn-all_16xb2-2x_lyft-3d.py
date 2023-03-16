_base_ = [
    '../_base_/models/pointpillars_hv_fpn_lyft.py',
    '../_base_/datasets/lyft-3d.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]
point_cloud_range = [-100, -100, -5, 100, 100, 3]
# Note that the order of class names should be consistent with
# the following anchors' order
class_names = [
    'bicycle', 'motorcycle', 'pedestrian', 'animal', 'car',
    'emergency_vehicle', 'bus', 'other_vehicle', 'truck'
]
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
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
train_dataloader = dict(
    batch_size=2, num_workers=4, dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# model settings
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=[-100, -100, -5, 100, 100, 3])),
    pts_voxel_encoder=dict(
        feat_channels=[32, 64],
        point_cloud_range=[-100, -100, -5, 100, 100, 3]),
    pts_middle_encoder=dict(output_shape=[800, 800]),
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        _delete_=True,
        type='ShapeAwareHead',
        num_classes=9,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGeneratorPerCls',
            ranges=[[-100, -100, -1.0709302, 100, 100, -1.0709302],
                    [-100, -100, -1.3220503, 100, 100, -1.3220503],
                    [-100, -100, -0.9122268, 100, 100, -0.9122268],
                    [-100, -100, -1.8012227, 100, 100, -1.8012227],
                    [-100, -100, -1.0715024, 100, 100, -1.0715024],
                    [-100, -100, -0.8871424, 100, 100, -0.8871424],
                    [-100, -100, -0.3519405, 100, 100, -0.3519405],
                    [-100, -100, -0.6276341, 100, 100, -0.6276341],
                    [-100, -100, -0.3033737, 100, 100, -0.3033737]],
            sizes=[
                [1.76, 0.63, 1.44],  # bicycle
                [2.35, 0.96, 1.59],  # motorcycle
                [0.80, 0.76, 1.76],  # pedestrian
                [0.73, 0.35, 0.50],  # animal
                [4.75, 1.92, 1.71],  # car
                [6.52, 2.42, 2.34],  # emergency vehicle
                [12.70, 2.92, 3.42],  # bus
                [8.17, 2.75, 3.20],  # other vehicle
                [10.24, 2.84, 3.44]  # truck
            ],
            custom_values=[],
            rotations=[0, 1.57],
            reshape_out=False),
        tasks=[
            dict(
                num_class=2,
                class_names=['bicycle', 'motorcycle'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=2,
                class_names=['pedestrian', 'animal'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=2,
                class_names=['car', 'emergency_vehicle'],
                shared_conv_channels=(64, 64, 64),
                shared_conv_strides=(2, 1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=3,
                class_names=['bus', 'other_vehicle', 'truck'],
                shared_conv_channels=(64, 64, 64),
                shared_conv_strides=(2, 1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01))
        ],
        assign_per_class=True,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            assigner=[
                dict(  # bicycle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # motorcycle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # animal
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # emergency vehicle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # other vehicle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
