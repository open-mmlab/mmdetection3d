# model settings
_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_rf2021_mvxfasterrcnn.py',    
    '../_base_/datasets/rf2021-3d-ped.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        point_cloud_range=[-60, -63.84, -3, 62.88, 60, 1]),
    pts_voxel_encoder=dict(
        point_cloud_range=[-60, -63.84, -3, 62.88, 60, 1]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[1, 1, 1],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 1, 1],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-60, -63.84, -0.0345, 62.88, 60, -0.0345]],
            sizes=[[0.7, 0.7, 1.7]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)))
