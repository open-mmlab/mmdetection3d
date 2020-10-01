_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-car.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# data settings
data = dict(train=dict(dataset=dict(load_interval=1)))

# model settings
model = dict(
    type='MVXFasterRCNN',
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345]],
            sizes=[[2.08, 4.73, 1.77]],
            rotations=[0, 1.57],
            reshape_out=True)))

# model training and testing settings
train_cfg = dict(
    _delete_=True,
    pts=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.55,
            neg_iou_thr=0.4,
            min_pos_iou=0.4,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False))
