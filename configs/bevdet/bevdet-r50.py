# Copyright (c) OpenMMLab. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

model = dict(
    type='BEVDet',
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='LSSFPN',
        in_channels=1024 + 2048,
        out_channels=512,
        upsampling_scale_output=None,
        input_feat_indexes=(0, 1),
        upsampling_scale=2,
        use_input_conv=True),
    view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=(256, 704),
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False),
    bev_encoder_backbone=dict(
        type='CustomResNet',
        depth=18,
        num_stages=3,
        stem_channels=64,
        base_channels=128,
        out_indices=(0, 1, 2),
        strides=(2, 2, 2),
        dilations=(1, 1, 1),
        frozen_stages=-1,
        with_cp=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    bev_encoder_neck=dict(
        type='LSSFPN', in_channels=64 * 8 + 64 * 2, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    # To avoid 'flip' information conflict between RandomFlip and RandomFlip3D,
    # 3D space augmentation should be conducted before loading images and
    # conducting image-view space augmentation.
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
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
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # The order of image-view augmentation should be
    # resize -> crop -> pad -> flip -> rotate
    dict(
        type='MultiViewWrapper',
        transforms=[
            dict(
                type='Resize', ratio_range=(0.864, 1.25),
                img_scale=(396, 704)),
            dict(
                type='RangeLimitedRandomCrop',
                relative_x_offset_range=(0.0, 1.0),
                relative_y_offset_range=(1.0, 1.0),
                crop_size=(256, 704)),
            dict(type='Pad', size=(256, 704)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='RandomRotate',
                range=(-5.4, 5.4),
                img_fill_val=0,
                level=1,
                prob=1.0),
            dict(type='Normalize', **img_norm_cfg)
        ],
        collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip',
                        'rotate']),
    dict(type='GetBEVDetInputs'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # The order of image-view augmentation should be
    # resize -> crop -> pad -> flip -> rotate
    dict(
        type='MultiViewWrapper',
        transforms=[
            dict(
                type='Resize',
                ratio_range=(1.091, 1.091),
                img_scale=(396, 704)),
            dict(
                type='RangeLimitedRandomCrop',
                relative_x_offset_range=(0.5, 0.5),
                relative_y_offset_range=(1.0, 1.0),
                crop_size=(256, 704)),
            dict(type='Pad', size=(256, 704)),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='RandomRotate',
                range=(-0.0, 0.0),
                img_fill_val=0,
                level=1,
                prob=0.0),
            dict(type='Normalize', **img_norm_cfg)
        ],
        collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip',
                        'rotate']),
    dict(type='GetBEVDetInputs'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        modality=input_modality),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        modality=input_modality))

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[19, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
