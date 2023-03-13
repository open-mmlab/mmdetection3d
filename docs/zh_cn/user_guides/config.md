# 学习配置文件

MMDetection3D 和其他 OpenMMLab 仓库使用 [MMEngine 的配置文件系统](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)。它具有模块化和继承性设计，以便于进行各种实验。

## 配置文件的内容

MMDetection3D 采用模块化设计，所有功能的模块可以通过配置文件进行配置。以 PointPillars 为例，我们将根据不同的功能模块介绍配置文件的各个字段。

### 模型配置

在 MMDetection3D 的配置中，我们使用 `model` 字段来配置检测算法的组件。除了 `voxel_encoder`，`backbone` 等神经网络组件外，还需要 `data_preprocessor`，`train_cfg` 和 `test_cfg`。`data_preprocessor` 负责对数据加载器（dataloader）输出的每一批数据进行预处理。模型配置中的 `train_cfg` 和 `test_cfg` 用于设置训练和测试组件的超参数。

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
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
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

### 数据集和评测器配置

在使用[执行器（Runner）](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)进行训练、测试和验证时，我们需要配置[数据加载器](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader)。构建数据加载器需要设置数据集和数据处理流程。由于这部分的配置较为复杂，我们使用中间变量来简化数据加载器配置的编写。

```python
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
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
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
```

[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html)用于计算训练模型在验证和测试数据集上的指标。评测器的配置由一个或一组评价指标配置组成：

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator
```

由于测试数据集没有标注文件，因此 MMDetection3D 中的 test_dataloader 和 test_evaluator 配置通常等于 val。如果您想要保存在测试数据集上的检测结果，则可以像这样编写配置：

```python
# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/velodyne_reduced'),
        ann_file='kitti_infos_test.pkl',
        load_eval_anns=False,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_test.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='results/kitti-3class/kitti_results')
```

### 训练和测试配置

MMEngine 的执行器使用循环（Loop）来控制训练，验证和测试过程。用户可以使用这些字段设置最大训练轮次和验证间隔：

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

### 优化配置

`optim_wrapper` 是配置优化相关设置的字段。优化器封装不仅提供了优化器的功能，还支持梯度裁剪、混合精度训练等功能。更多内容请看[优化器封装教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html)。

```python
optim_wrapper = dict(  # 优化器封装配置
    type='OptimWrapper',  # 优化器封装类型，切换到 AmpOptimWrapper 启动混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器，请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))  # 梯度裁剪选项。设置为 None 禁用梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html
```

`param_scheduler` 是配置调整优化器超参数（例如学习率和动量）的字段。用户可以组合多个调度器来创建所需要的参数调整策略。更多信息请参考[参数调度器教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)和[参数调度器 API 文档](https://mmengine.readthedocs.io/zh_CN/latest/api/optim.html#scheduler)。

```python
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=32,
        eta_min=0.01,
        begin=0,
        end=32,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=48,
        eta_min=1.0000000000000001e-07,
        begin=32,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=32,
        eta_min=0.8947368421052632,
        begin=0,
        end=32,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=48,
        eta_min=1,
        begin=32,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
]
```

### 钩子配置

用户可以在训练、验证和测试循环上添加钩子，从而在运行期间插入一些操作。有两种不同的钩子字段，一种是 `default_hooks`，另一种是 `custom_hooks`。

`default_hooks` 是一个钩子配置字典，并且这些钩子是运行时所需要的。它们具有默认优先级，是不需要修改的。如果未设置，执行器将使用默认值。如果要禁用默认钩子，用户可以将其配置设置为 `None`。

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
```

`custom_hooks` 是一个由其他钩子配置组成的列表。用户可以开发自己的钩子并将其插入到该字段中。

```python
custom_hooks = []
```

### 运行配置

```python
default_scope = 'mmdet3d'  # 寻找模块的默认注册器域。请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # 是否启用 cudnn benchmark
    mp_cfg=dict(  # 多进程配置
        mp_start_method='fork',  # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能不安全。请参考 https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # 关闭 opencv 的多进程以避免系统超负荷
    dist_cfg=dict(backend='nccl'))  # 分布式配置

vis_backends = [dict(type='LocalVisBackend')]  # 可视化后端。请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(
    type='LogProcessor',  # 日志处理器用于处理运行时日志
    window_size=50,  # 日志数值的平滑窗口
    by_epoch=True)  # 是否使用 epoch 格式的日志。需要与训练循环的类型保持一致

log_level = 'INFO'  # 日志等级
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。如果 `load_from` 为 None，它将恢复 `work_dir` 中的最近检查点。
```

## 配置文件继承

在 `configs/_base_` 文件夹下有 4 个基本组件类型，分别是：数据集（dataset），模型（model），训练策略（schedule）和运行时的默认设置（default runtime）。许多方法，如 SECOND、PointPillars、PartA2、VoteNet 都能够很容易地构建出来。由 `_base_` 下的组件组成的配置，被我们称为 _原始配置（primitive）_。

对于同一个文件夹下的所有配置，推荐**只有一个**对应的 _原始配置_ 文件。所有其他的配置文件都应该继承自这个 _原始配置_ 文件。这样就能保证配置文件的最大继承深度为 3。

为了便于理解，我们建议贡献者继承现有方法。例如，如果在 PointPillars 的基础上做了一些修改，用户可以首先通过指定 `_base_ = '../pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'` 来继承基础的 PointPillars 结构，然后修改配置文件中的必要参数以完成继承。

如果您在构建一个与任何现有方法都不共享的全新方法，那么可以在 `configs` 文件夹下创建一个新的例如 `xxx_rcnn` 文件夹。

更多细节请参考 [MMEngine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)。

通过设置 `_base_` 字段，我们可以设置当前配置文件继承自哪些文件。

当 `_base_` 为文件路径字符串时，表示继承一个配置文件的内容。

```python
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
```

当 `_base_` 是多个文件路径组成的列表式，表示继承多个文件。

```python
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
```

如果需要检测配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。

### 忽略基础配置文件里的部分字段

有时，您也许会设置 `_delete_=True` 去忽略基础配置文件里的一些字段。您可以参考 [MMEngine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html) 来获得一些简单的指导。

在 MMDetection3D 里，例如，修改以下 PointPillars 配置中的颈部网络：

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

`FPN` 和 `SECONDFPN` 使用不同的关键字来构建：

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

`_delete_=True` 将使用新的键去替换 `pts_neck` 字段内所有旧的键。

### 在配置文件里使用中间变量

配置文件里会使用一些中间变量，例如数据集里的 `train_pipeline`/`test_pipeline`。需要注意的是，当修改子配置文件中的中间变量时，用户需要再次将中间变量传递到对应的字段中。例如，我们想使用多尺度策略训练并测试 PointPillars，`train_pipeline`/`test_pipeline` 是我们想要修改的中间变量。

```python
_base_ = './nus-3d.py'
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        backend_args=backend_args),
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
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        backend_args=backend_args),
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
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

我们首先定义新的 `train_pipeline`/`test_pipeline`，然后传递到数据加载器字段中。

### 复用 \_base\_ 文件中的变量

如果用户希望复用 base 文件中的变量，则可以通过使用 `{{_base_.xxx}}` 获取对应变量的拷贝。例如：

```python
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'

a = {{_base_.model}}  # 变量 `a` 等于 `_base_` 中定义的 `model`
```

## 通过脚本参数修改配置

当使用 `tools/train.py` 或者 `tools/test.py` 提交工作时，您可以通过指定 `--cfg-options` 来修改配置文件。

- 更新配置字典的键值

  可以按照原始配置文件中字典的键值顺序指定配置选项。例如，使用 `--cfg-options model.backbone.norm_eval=False` 将模型主干网络中的所有 BN 模块都改为 `train` 模式。

- 更新配置列表中的键值

  在配置文件里，一些配置字典被包含在列表中，例如，训练流程 `train_dataloader.dataset.pipeline` 通常是一个列表，例如 `[dict(type='LoadPointsFromFile'), ...]`。如果您想要将训练流程中的 `'LoadPointsFromFile'` 改成 `'LoadPointsFromDict'`，您需要指定 `--cfg-options data.train.pipeline.0.type=LoadPointsFromDict`。

- 更新列表/元组的值

  如果要更新的值是列表或元组。例如，配置文件通常设置 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`。如果您想要改变这个均值，您需要指定 `--cfg-options model.data_preprocessor.mean="[127,127,127]"`。注意，引号 `"` 是支持列表/元组数据类型所必需的，并且在指定值的引号内**不允许**有空格。

## 配置文件名称风格

我们遵循以下样式来命名配置文件。建议贡献者遵循相同的风格。

```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}.py
```

文件名分为五个部分。所有部分和组件用 `_` 连接，每个部分或组件内的单词应该用 `-` 连接。

- `{algorithm name}`：算法的名称。它可以是检测器的名称，例如 `pointpillars`、`fcos3d` 等。
- `{model component names}`：算法中使用的组件名称，如 voxel_encoder、backbone、neck 等。例如 `second_secfpn_head-dcn-circlenms` 表示使用 SECOND 的 SparseEncoder，SECONDFPN，以及带有 DCN 和 circle NMS 的检测头。
- `{training settings}`：训练设置的信息，例如批量大小，数据增强，损失函数策略，调度器以及训练轮次/迭代。例如 `8xb4-tta-cyclic-20e` 表示使用 8 个 gpu，每个 gpu 有 4 个数据样本，测试增强，余弦退火学习率，训练 20 个 epoch。缩写介绍：
  - `{gpu x batch_per_gpu}`：GPU 数和每个 GPU 的样本数。`bN` 表示每个 GPU 上的批量大小为 N。例如 `4xb4` 是 4 个 GPU，每个 GPU 有 4 个样本数的缩写。
  - `{schedule}`：训练方案，可选项为 `schedule-2x`、`schedule-3x`、`cyclic-20e` 等。`schedule-2x` 和 `schedule-3x` 分别代表 24 epoch 和 36 epoch。`cyclic-20e` 表示 20 epoch。
- `{training dataset information}`：训练数据集名，例如 `kitti-3d-3class`，`nus-3d`，`s3dis-seg`，`scannet-seg`，`waymoD5-3d-car`。这里 `3d` 表示数据集用于 3D 目标检测，`seg` 表示数据集用于点云分割。
- `{testing dataset information}`（可选）：当模型在一个数据集上训练，在另一个数据集上测试时的测试数据集名。如果没有注明，则表示训练和测试的数据集类型相同。
