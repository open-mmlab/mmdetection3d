# 学习配置文件

MMDetection3D 和其他 OpenMMLab 仓库使用[MMEngine 配置系统](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html)。它具有模块化和继承性设计，便于进行各种实验。如果希望检查配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。

## 配置文件内容

MMDetection3D 使用模块化的设计，所有具有不同功能的模块都可以通过配置文件进行配置。以 PointPillars 为例，我们将根据不同的功能模块介绍配置文件的每一个字段。

### 模型配置

在 mmdetection3d 配置中，我们使用 `model` 设置检测算法组件。除了神经网络组件，如 `voxel_encoder`，`backbone` 等，它还需要 `data_preprocessor`，`train_cfg` 和 `test_cfg`。`data_preprocessor` 负责处理 dataloader 输出的一个批次的数据。模型配置中的 `train_cfg` 和 `test_cfg` 用于设置训练和测试组件的超参数。

```python
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            voxel_size=[0.16, 0.16, 4],
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
```

### 数据集和评估器配置

[执行器（Runner）](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)的训练，验证和测试需要[数据加载器（dataloaders）](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader)。构建数据加载器需要数据集和数据流水线。由于这一部分的复杂性，我们使用中间变量简化数据加载配置文件的编写。

```python
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=['Pedestrian', 'Cyclist', 'Car'])
db_sampler = dict(
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=['Pedestrian', 'Cyclist', 'Car'],
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4))
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='data/kitti/',
            info_path='data/kitti/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4)),
        use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
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
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1])
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='data/kitti/',
                        info_path='data/kitti/kitti_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                Car=5, Pedestrian=5, Cyclist=5)),
                        classes=['Pedestrian', 'Cyclist', 'Car'],
                        sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
                        points_loader=dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4)),
                    use_ground_plane=True),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
                dict(type='PointShuffle'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1])
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
        box_type_3d='LiDAR'))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1])
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
        box_type_3d='LiDAR'))
```

[评估器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/metric_and_evaluator.html)用来计算训练模型在验证集和测试集上的评价指标。评估器的配置包含一个或一系列指标配置：

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox')
```

### 训练和测试配置

MMEngine 的执行器使用循环（Loop）来控制训练、验证以及测试过程。用户可以用以下字段设置最大训练周期以及验证间隔。

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

### 优化器配置

`optim_wrapper` 字段用来配置优化器相关设置。优化器包装器不仅提供优化器的功能，用时也支持其它功能，如梯度裁剪、混合精度训练等。更多细节请参考[优化器封装教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optimizer.html)。

```python
optim_wrapper = dict(  # 优化器封装配置
    type='OptimWrapper',  # 优化器封装类型，切换成 AmpOptimWrapper 使用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 中所有类型的优化器。参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))  # 梯度裁剪选项。设置 None 禁用梯度裁剪。用法请参考 https://mmengine.readthedocs.io/zh_CN/latest/tutorials
```

`param_scheduler` 字段用来配置调整优化器超参数，例如学习率和动量。用户可以组合多个调度器来创建所需要的参数调整策略。更多细节请参考[参数调度器教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)和[参数调度器 API 文档](TODO)。

```python
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=32.0,
        eta_min=0.01,
        begin=0,
        end=32.0,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=48.0,
        eta_min=1.0000000000000001e-07,
        begin=32.0,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=32.0,
        eta_min=0.8947368421052632,
        begin=0,
        end=32.0,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=48.0,
        eta_min=1,
        begin=32.0,
        end=80,
        convert_to_iter_based=True)
]
```

### 钩子配置

用户可以将钩子连接到训练、验证和测试中，以便在运行期间插入一些操作。有两个不同的钩子字段：`default_hooks` 和 `custom_hooks`。

`default_hooks` 是一个钩子配置字典。`default_hooks` 里的钩子是在运行时所需要的。它们有默认的优先级，是不需要修改的。如果没有设置，执行器会使用默认值。如果需要禁用默认的钩子，用户可以将其配置设置成 `None`。

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))
```

### 运行配置

```python
default_scope = 'mmdet3d'  # 寻找模块的默认注册域。参考 https://mmengine.readthedocs.io/zh_CN/latest/tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # 是否使用 cudnn benchmark
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 使用 fork 开启多线程。'fork' 通常比 'spawn' 快，但可能不安全。可参考 https://github.com/pytorch/pytorch/issues/1355
    dist_cfg=dict(backend='nccl'))  # 分布式配置
vis_backends = [dict(type='LocalVisBackend')]  # 可视化后端
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
```

## 配置文件继承性

在 `configs/_base_` 文件夹下有 4 个基本组件类型，分别是：数据集（dataset），模型（model），训练策略（schedule）和运行时的默认设置（default runtime）。通过从上述文件夹中选取一个组件进行组合，许多方法如 SECOND、PointPillars、PartA2 和 VoteNet 都能够很容易地构建出来。由 `_base_` 下的组件组成的配置，被我们称为 _原始配置(primitive)_。

对于同一文件夹下的配置，推荐**只有一个**对应的 _原始配置_ 文件，所有其他的配置文件都应该继承自这个 _原始配置_ 文件，这样就能保证配置文件的最大继承深度为 3。

为了便于理解，我们建议贡献者继承现有方法。例如，如果在 PointPillars 的基础上做了一些修改，用户首先可以通过指定 `_base_ = ../pointpillars/pointpillars_hv_fpn_sbn-all_8xb4_2x_nus-3d.py` 来继承基础的 PointPillars 结构，然后修改配置文件中的必要参数以完成继承。

如果你在构建一个与任何现有方法不共享结构的全新方法，可以在 `configs` 文件夹下创建一个新的例如 `xxx_rcnn` 文件夹。

更多细节请参考 [mmengine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html)。

### 忽略基础配置中的某些字段

有时候，你需要设置 `_delete_=True` 来忽略基础配置中的某些字段。你可以参考 [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) 做简单了解。

在 MMDetection3D 中，例如，修改以下 PointPillars 配置中的 FPN 瓶颈网络。

```python
model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(voxel_layer=dict(...)),
    pts_voxel_encoder=dict(...),
    pts_middle_encoder=dict(...),
    pts_backbone=dict(...),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(...))
```

`FPN` 和 `SECONDFPN` 使用不同的关键字来构建。

```python
_base_ = '../_base_/models/pointpillars_hv_fpn_nus.py'
model = dict(
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(...))
```

`_delete_=True` 将会使用新的键值替换 `pts_neck` 中的旧键值。

### 在配置中使用中间变量

在配置文件中通常会使用一些中间变量，例如数据集中 `train_pipeline`/`test_pipeline`。需要注意的是当在子配置中修改中间变量，用户需要再次将中间变量传递到相应的字段中。例如，我们想要使用多尺度策略训练和测试 PointPillars。`train_pipeline`/`test_pipeline` 是我们需要修改的中间变量。

```python
_base_ = './nus-3d.py'
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=[0.95, 1.0, 1.05],
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
      dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

### 重使用 \_base\_ 文件中的变量

如果用户想重新使用基础文件中的变量，可以通过使用 `{{_base_.xxx}}` 拷贝相应的变量。例如：

```python
_base_ = './pointpillars_hv_secfpn_8xb6_160e_kitti-3d-3class.py'

a = {{_base_.model}} # 变量 `a` 和 `_base_` 中定义的 `model` 相同
```

### 通过脚本参数修改配置

当使用 "tools/train.py" 或者 "tools/test.py" 提交工作任务时，可以指定 `--cfg-options` 来修改配置。

- 更新配置字典的键值

  可以按照原始配置中字典的键值顺序指定配置选项。例如，`--cfg-options model.backbone.norm_eval=False` 改变模型骨干网络中的 BN 模块为 `train` 模式。

- 更新配置列表中的键值

  配置中一些配置字典是由列表组成。例如，训练流水线 `train_dataloader.dataset.pipeline` 通常是一个列表，例如 `[dict(type='LoadImageFromFile'), ...]`。如果你想将流水线中的 `LoadImageFromFile` 改为 `LoadImageFromNDArray`，你可以指定 `--cfg-options data.train.pipeline.0.type=LoadImageFromNDArray`。

- 更新列表/元组值

  如果更新的值是列表或元组。例如，配置文件中通常设置 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`。如果你想改变这个均值，你可以指定 `--cfg-options model.data_preprocessor.mean="[127,127,127]"`。注意引用符号 `"` 是用来支持列表/元组数据类型所必要的，并且在指定值的引用符号内**没有**空格。

## 配置文件名称风格

我们遵循以下样式来命名配置文件，并建议贡献者遵循相同的风格。

```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}.py
```

文件名分为五个部分。所有部分和组件通过 `_` 连接，每个部分或组件单词用 `-` 连接。

- `{algorithm name}`：算法名。这应该是检测器的名字例如 `pointpillars`，`fcos3d` 等。
- `{model component names}`：算法中使用的组件名，例如体素编码器，骨干网络，瓶颈网络等。例如 `second_secfpn_head-dcn-circlenms` 意味着使用 SECOND's SparseEncoder，SECONDFPN 以及使用 DCN 和 circle NMS 的检测头。
- `{training settings}`：训练设置信息，例如批量大小，数据增强，损失函数策略，调度器以及 epoch/迭代等。例如：`8xb4-tta-cyclic-20e` 意味着使用 8 个 GPUs，每个 GPU 有 4 个数据样本，测试增强，余弦退火学习率以及训练 20 个 epoch。一些缩写：
  - `{gpu x batch_per_gpu}`：GPUs 数以及每块 GPU 的样本数。`bN` 表示 每块 GPU 的批量大小为 N。例如 `4xb4` 是 4-gpus x 4-samples-per-gpu 的简写。
  - `{schedule}`：训练调度，可选项为 `schedule-2x`，`schedule-3x`，`cyclic-20e`等。`schedule-2x` 和 `schedule-3x` 分别表示训练 24 和 36 个 epoch。`cyclic-20e` 表示训练 20 个 epoch。
- `{training dataset information}`：训练数据集名如 `kitti-3d-3class`，`nus-3d`，`s3dis-seg`，`scannet-seg`，`waymoD5-3d-car`。此处 `3d` 表示数据集用于 3d 目标检测，`seg` 表示数据集用于点云分割。
- `{testing dataset information}`（可选）：当模型在一个数据集上训练，在另一个数据集上测试时的测试数据集名。如果没有指定，意味着模型在同一数据类型上训练和测试。
