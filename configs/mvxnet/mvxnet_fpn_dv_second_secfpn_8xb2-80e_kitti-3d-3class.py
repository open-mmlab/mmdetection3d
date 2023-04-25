_base_ = [
    '../_base_/datasets/kitti-3d-3class.py', '../_base_/models/mvxnet.py',
    '../_base_/schedules/cosine.py', '../_base_/default_runtime.py'
]

# dataset settings
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(pts='training/velodyne_reduced', img='training/image_2')
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='RandomResize', scale=[(640, 192), (2560, 768)], keep_ratio=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=0, keep_ratio=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        dataset=dict(
            data_prefix=data_prefix,
            pipeline=train_pipeline,
            modality=input_modality,
            filter_empty_gt=False)))
val_dataloader = dict(
    dataset=dict(
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality))
test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# You may need to download the model first is the network is unstable
load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'  # noqa
