# Full MVX-Net config: MobileNetV2+FPN (camera) + RPFNet (LiDAR).
# KITTI 3-class (Car, Pedestrian, Cyclist)

_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type='DynamicMVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(max_num_points=-1, point_cloud_range=point_cloud_range,
                         voxel_size=voxel_size, max_voxels=(-1, -1)),
        mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False, pad_size_divisor=32),

    # ----------------------- image branch -----------------------
    img_backbone=dict(
        type='mmdet.MobileNetV2',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[16, 24, 32, 64],
        out_channels=256,
        num_outs=4,
        norm_cfg=dict(type='BN', requires_grad=False)),

    # ----------------------- LiDAR voxel encoder ----------------
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type='PointFusion', img_channels=256, pts_channels=64,
            mid_channels=128, out_channels=128, img_levels=[0,1,2,3],
            align_corners=False, activate_out=True, fuse_out=False)),

    # ----------------------- Sparse middle encoder --------------
    pts_middle_encoder=dict(
        type='SparseEncoder', in_channels=128,
        sparse_shape=[41, 1600, 1408], order=('conv', 'norm', 'act')),

    # ----------------------- FireRPFNet backbone --------------------
    pts_backbone=dict(
       # type='RPFNet',
        type='FireRPFNet',
        in_channels=256,
        layer_channels=[128, 256, 256, 256], with_cbam=True),
    pts_neck=None,

    # ----------------------- Anchor head ------------------------
    pts_bbox_head=dict(
        type='Anchor3DHead', num_classes=3,
        in_channels=256, feat_channels=256, use_direction_classifier=True,
        anchor_generator=dict(type='Anchor3DRangeGenerator',
            ranges=[[0,-40,-0.6,70.4,40,-0.6],[0,-40,-0.6,70.4,40,-0.6],
                    [0,-40,-1.78,70.4,40,-1.78]],
            sizes=[[0.8,0.6,1.73],[1.76,0.6,1.73],[3.9,1.6,1.56]],
            rotations=[0,1.57], reshape_out=False),
        assigner_per_size=True, diff_rad_by_sin=True, assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0/9.0,
                       loss_weight=2.0),
        loss_dir=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False,
                      loss_weight=0.2)),

    train_cfg=dict(
        pts=dict(assigner=[
            dict(type='Max3DIoUAssigner', iou_calculator=dict(type='BboxOverlapsNearest3D'),
                 pos_iou_thr=0.35, neg_iou_thr=0.2, min_pos_iou=0.2,
                 ignore_iof_thr=-1),
            dict(type='Max3DIoUAssigner', iou_calculator=dict(type='BboxOverlapsNearest3D'),
                 pos_iou_thr=0.35, neg_iou_thr=0.2, min_pos_iou=0.2,
                 ignore_iof_thr=-1),
            dict(type='Max3DIoUAssigner', iou_calculator=dict(type='BboxOverlapsNearest3D'),
                 pos_iou_thr=0.6, neg_iou_thr=0.45, min_pos_iou=0.45,
                 ignore_iof_thr=-1)], allowed_border=0, pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(use_rotate_nms=True, nms_across_levels=False, nms_thr=0.01,
                  score_thr=0.1, min_bbox_size=0, nms_pre=100, max_num=50))
)

# -----------------------------------------------------------------------------
# Dataset & pipelines (unchanged)
# -----------------------------------------------------------------------------

dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True),
    dict(type='RandomResize', scale=[(320, 96), (1280, 384)], keep_ratio=True),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816,0.78539816],
         scale_ratio_range=[0.95,1.05], translation_std=[0.2,0.2,0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points','img','gt_bboxes_3d','gt_labels_3d',
                                       'gt_bboxes','gt_labels'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='MultiScaleFlipAug3D', img_scale=(1280,384), pts_scale_ratio=1,
         flip=False, transforms=[
            dict(type='Resize', scale=0, keep_ratio=True),
            dict(type='GlobalRotScaleTrans', rot_range=[0,0], scale_ratio_range=[1.,1.],
                 translation_std=[0,0,0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
         ]),
    dict(type='Pack3DDetInputs', keys=['points','img'])
]

modality = dict(use_lidar=True, use_camera=True)

train_dataloader = dict(
    batch_size=2, num_workers=4, sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset', times=2, dataset=dict(
        type=dataset_type, data_root=data_root, modality=modality,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=train_pipeline, filter_empty_gt=False, metainfo=metainfo,
        box_type_3d='LiDAR', backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1, num_workers=1, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, modality=modality,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline, metainfo=metainfo, test_mode=True,
        box_type_3d='LiDAR', backend_args=backend_args))

test_dataloader = val_dataloader

# -----------------------------------------------------------------------------
# Optimizer / runtime
# -----------------------------------------------------------------------------
optim_wrapper = dict(optimizer=dict(lr=0.001, weight_decay=0.01),
                     clip_grad=dict(max_norm=35, norm_type=2))

val_evaluator = dict(type='KittiMetric', ann_file='data/kitti/kitti_infos_val.pkl')

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=5)
