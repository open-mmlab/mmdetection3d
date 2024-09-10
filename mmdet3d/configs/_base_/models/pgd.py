# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from .fcos3d import *

from mmdet3d.models.dense_heads.pgd_head import PGDHead
from mmdet3d.models.task_modules.coders.pgd_bbox_coder import PGDBBoxCoder

# model settings
model.merge(
    dict(
        bbox_head=dict(
            _delete_=True,
            type=PGDHead,
            num_classes=10,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            use_direction_classifier=True,
            diff_rad_by_sin=True,
            pred_attrs=True,
            pred_velo=True,
            pred_bbox2d=True,
            pred_keypoints=False,
            dir_offset=0.7854,  # pi/4
            strides=[8, 16, 32, 64, 128],
            group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo
            cls_branch=(256, ),
            reg_branch=(
                (256, ),  # offset
                (256, ),  # depth
                (256, ),  # size
                (256, ),  # rot
                ()  # velo
            ),
            dir_branch=(256, ),
            attr_branch=(256, ),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_attr=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            norm_on_bbox=True,
            centerness_on_reg=True,
            center_sampling=True,
            conv_bias=True,
            dcn_on_last_conv=True,
            use_depth_classifier=True,
            depth_branch=(256, ),
            depth_range=(0, 50),
            depth_unit=10,
            division='uniform',
            depth_bins=6,
            bbox_coder=dict(type=PGDBBoxCoder, code_size=9)),
        test_cfg=dict(
            nms_pre=1000, nms_thr=0.8, score_thr=0.01, max_per_img=200)))
