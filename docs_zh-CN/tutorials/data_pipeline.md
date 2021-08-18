# 教程 3: 自定义数据预处理流程

## 数据预处理流程的设计

遵循一般惯例，我们使用 `Dataset` 和 `DataLoader` 来调用多个进程进行数据的加载。`Dataset` 将会返回与模型前向传播的参数所对应的数据项构成的字典。因为目标检测中的数据的尺寸可能无法保持一致（如点云中点的数量、真实标注框的尺寸等），我们在 MMCV 中引入一个 `DataContainer` 类型，来帮助收集和分发不同尺寸的数据。请参考[此处](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)获取更多细节。

数据预处理流程和数据集之间是互相分离的两个部分，通常数据集定义了如何处理标注信息，而数据预处理流程定义了准备数据项字典的所有步骤。数据集预处理流程包含一系列的操作，每个操作将一个字典作为输入，并输出应用于下一个转换的一个新的字典。

我们将在下图中展示一个最经典的数据集预处理流程，其中蓝色框表示预处理流程中的各项操作。随着预处理的进行，每一个操作都会添加新的键值（图中标记为绿色）到输出字典中，或者更新当前存在的键值（图中标记为橙色）。
![](https://github.com/open-mmlab/mmdetection3d/blob/master/resources/data_pipeline.png)

预处理流程中的各项操作主要分为数据加载、预处理、格式化、测试时的数据增强。

接下来将展示一个用于 PointPillars 模型的数据集预处理流程的例子。

```python
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        pts_scale_ratio=1.0,
        flip=False,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
```

对于每项操作，我们将列出相关的被添加/更新/移除的字典项。

### 数据加载

`LoadPointsFromFile`
- 添加：points

`LoadPointsFromMultiSweeps`
- 更新：points

`LoadAnnotations3D`
- 添加：gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, pts_instance_mask, pts_semantic_mask, bbox3d_fields, pts_mask_fields, pts_seg_fields

### 预处理

`GlobalRotScaleTrans`
- 添加：pcd_trans, pcd_rotation, pcd_scale_factor
- 更新：points, *bbox3d_fields

`RandomFlip3D`
- 添加：flip, pcd_horizontal_flip, pcd_vertical_flip
- 更新：points, *bbox3d_fields

`PointsRangeFilter`
- 更新：points

`ObjectRangeFilter`
- 更新：gt_bboxes_3d, gt_labels_3d

`ObjectNameFilter`
- 更新：gt_bboxes_3d, gt_labels_3d

`PointShuffle`
- 更新：points

`PointsRangeFilter`
- 更新：points

### 格式化

`DefaultFormatBundle3D`
- 更新：points, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels

`Collect3D`
- 添加：img_meta （由 `meta_keys` 指定的键值构成的 img_meta）
- 移除：所有除 `keys` 指定的键值以外的其他键值

### 测试时的数据增强

`MultiScaleFlipAug`
- 更新: scale, pcd_scale_factor, flip, flip_direction, pcd_horizontal_flip, pcd_vertical_flip （与这些指定的参数对应的增强后的数据列表）

## 扩展并使用自定义数据集预处理方法

1. 在任意文件中写入新的数据集预处理方法，如 `my_pipeline.py`，该预处理方法的输入和输出均为字典

    ```python
    from mmdet.datasets import PIPELINES

    @PIPELINES.register_module()
    class MyTransform:

        def __call__(self, results):
            results['dummy'] = True
            return results
    ```

2. 导入新的预处理方法类

    ```python
    from .my_pipeline import MyTransform
    ```

3. 在配置文件中使用该数据集预处理方法

    ```python
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
        dict(type='MyTransform'),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    ```
