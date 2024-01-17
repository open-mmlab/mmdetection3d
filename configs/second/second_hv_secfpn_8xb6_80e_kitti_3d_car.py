# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.kitti_3d_car import *
    from .._base_.default_runtime import *
    from .._base_.models.second_hv_secfpn_kitti import *
    from .._base_.schedules.cyclic_40e import *

point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model.update(
    bbox_head=dict(
        type=Anchor3DHead,
        num_classes=1,
        anchor_generator=dict(
            type=Anchor3DRangeGenerator,
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),

    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=Max3DIoUAssigner,
            iou_calculator=dict(type=BboxOverlapsNearest3D),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False))
