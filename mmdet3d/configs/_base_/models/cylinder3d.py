# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models import Cylinder3D
from mmdet3d.models.backbones import Asymm3DSpconv
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.decode_heads.cylinder3d_head import Cylinder3DHead
from mmdet3d.models.losses import LovaszLoss
from mmdet3d.models.voxel_encoders import SegVFE

grid_shape = [480, 360, 32]
point_cloud_range = [0, -3.14159265359, -4, 50, 3.14159265359, 2]
model = dict(
    type=Cylinder3D,
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type=SegVFE,
        in_channels=6,
        feat_channels=[64, 128, 256, 256],
        with_voxel_center=True,
        grid_shape=grid_shape,
        point_cloud_range=point_cloud_range,
        feat_compression=16),
    backbone=dict(
        type=Asymm3DSpconv,
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type=Cylinder3DHead,
        channels=128,
        num_classes=20,
        dropout_ratio=0,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(type=LovaszLoss, loss_weight=1.0, reduction='none'),
        conv_seg_kernel_size=3,
        ignore_index=19),
    train_cfg=None,
    test_cfg=dict(mode='whole'))
