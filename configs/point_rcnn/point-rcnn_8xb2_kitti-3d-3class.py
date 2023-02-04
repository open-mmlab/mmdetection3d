_base_ = [
    '../_base_/datasets/kitti-3d-car.py', '../_base_/models/point_rcnn.py',
    '../_base_/default_runtime.py', '../_base_/schedules/cyclic-40e.py'
]

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    sample_groups=dict(Car=20, Pedestrian=15, Cyclist=15),
    classes=class_names,
    points_loader=dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=16384, sample_range=40.0),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384, sample_range=40.0)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))

lr = 0.001  # max learning rate
optim_wrapper = dict(optimizer=dict(lr=lr, betas=(0.95, 0.85)))
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
param_scheduler = [
    # learning rate scheduler
    # During the first 35 epochs, learning rate increases from 0 to lr * 10
    # during the next 45 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=35,
        eta_min=lr * 10,
        begin=0,
        end=35,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        eta_min=lr * 1e-4,
        begin=35,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 35 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 45 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=35,
        eta_min=0.85 / 0.95,
        begin=0,
        end=35,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=45,
        eta_min=1,
        begin=35,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True)
]
