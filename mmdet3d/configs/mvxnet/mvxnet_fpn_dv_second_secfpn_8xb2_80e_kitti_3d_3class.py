# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.schedules.cosine import *
    from .._base_.default_runtime import *

from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import RandomResize, Resize
from mmengine.dataset.dataset_wrapper import RepeatDataset
from mmengine.dataset.sampler import DefaultSampler
from mmengine.visualization.vis_backend import LocalVisBackend

from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile)
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.transforms.transforms_3d import (GlobalRotScaleTrans,
                                                       ObjectRangeFilter,
                                                       PointShuffle,
                                                       PointsRangeFilter,
                                                       RandomFlip3D)
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric
from mmdet3d.models.backbones.second import SECOND
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.models.detectors.mvx_faster_rcnn import DynamicMVXFasterRCNN
from mmdet3d.models.layers.fusion_layers.point_fusion import PointFusion
from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
from mmdet3d.models.necks.second_fpn import SECONDFPN
from mmdet3d.models.task_modules.anchor.anchor_3d_generator import \
    Anchor3DRangeGenerator
from mmdet3d.models.task_modules.assigners.max_3d_iou_assigner import \
    Max3DIoUAssigner
from mmdet3d.models.task_modules.coders.delta_xyzwhlr_bbox_coder import \
    DeltaXYZWLHRBBoxCoder
from mmdet3d.models.voxel_encoders.voxel_encoder import DynamicVFE
from mmdet3d.structures.ops.iou3d_calculator import BboxOverlapsNearest3D
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    type=DynamicMVXFasterRCNN,
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)),
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # make the image features more stable numerically to avoid loss nan
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    pts_voxel_encoder=dict(
        type=DynamicVFE,
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type=PointFusion,
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False)),
    pts_middle_encoder=dict(
        type=SparseEncoder,
        in_channels=128,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    pts_backbone=dict(
        type=SECOND,
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    pts_neck=dict(
        type=SECONDFPN,
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    pts_bbox_head=dict(
        type=Anchor3DHead,
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type=Anchor3DRangeGenerator,
            ranges=[
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type=DeltaXYZWLHRBBoxCoder),
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
        pts=dict(
            assigner=[
                dict(  # for Pedestrian
                    type=Max3DIoUAssigner,
                    iou_calculator=dict(type=BboxOverlapsNearest3D),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type=Max3DIoUAssigner,
                    iou_calculator=dict(type=BboxOverlapsNearest3D),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type=Max3DIoUAssigner,
                    iou_calculator=dict(type=BboxOverlapsNearest3D),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True),
    dict(type=RandomResize, scale=[(640, 192), (2560, 768)], keep_ratio=True),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type=RandomFlip3D, flip_ratio_bev_horizontal=0.5),
    dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=PointShuffle),
    dict(
        type=Pack3DDetInputs,
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ])
]
test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=MultiScaleFlipAug3D,
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # Temporary solution, fix this after refactor the augtest
            dict(type=Resize, scale=0, keep_ratio=True),
            dict(
                type=GlobalRotScaleTrans,
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type=RandomFlip3D),
            dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
        ]),
    dict(type=Pack3DDetInputs, keys=['points', 'img'])
]
modality = dict(use_lidar=True, use_camera=True)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=RepeatDataset,
        times=2,
        dataset=dict(
            type=KittiDataset,
            data_root=data_root,
            modality=modality,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=KittiDataset,
        data_root=data_root,
        modality=modality,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=KittiDataset,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        modality=modality,
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

optim_wrapper.update(
    dict(
        optimizer=dict(weight_decay=0.01),
        clip_grad=dict(max_norm=35, norm_type=2),
    ))
val_evaluator = dict(
    type=KittiMetric, ann_file='data/kitti/kitti_infos_val.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# You may need to download the model first is the network is unstable
load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'  # noqa
