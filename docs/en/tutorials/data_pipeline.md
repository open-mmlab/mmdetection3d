# Tutorial 3: Customize Data Pipelines

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. `Dataset` returns a dict of data items corresponding
the arguments of models' forward method.
Since the data in object detection may not be the same size (point number, gt bbox size, etc.),
we introduce a new `DataContainer` type in MMCV to help collect and distribute
data of different size.
See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

We present a classical pipeline in the following figure. The blue blocks are pipeline operations. With the pipeline going on, each operator can add new keys (marked as green) to the result dict or update the existing keys (marked as orange).
![](https://github.com/open-mmlab/mmdetection3d/blob/master/resources/data_pipeline.png)

The operations are categorized into data loading, pre-processing, formatting and test-time augmentation.

Here is an pipeline example for PointPillars.

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

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`LoadPointsFromFile`
- add: points

`LoadPointsFromMultiSweeps`
- update: points

`LoadAnnotations3D`
- add: gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, pts_instance_mask, pts_semantic_mask, bbox3d_fields, pts_mask_fields, pts_seg_fields

### Pre-processing

`GlobalRotScaleTrans`
- add: pcd_trans, pcd_rotation, pcd_scale_factor
- update: points, *bbox3d_fields

`RandomFlip3D`
- add: flip, pcd_horizontal_flip, pcd_vertical_flip
- update: points, *bbox3d_fields

`PointsRangeFilter`
- update: points

`ObjectRangeFilter`
- update: gt_bboxes_3d, gt_labels_3d

`ObjectNameFilter`
- update: gt_bboxes_3d, gt_labels_3d

`PointShuffle`
- update: points

`PointsRangeFilter`
- update: points

### Formatting

`DefaultFormatBundle3D`
- update: points, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels

`Collect3D`
- add: img_meta (the keys of img_meta is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

### Test time augmentation

`MultiScaleFlipAug`
- update: scale, pcd_scale_factor, flip, flip_direction, pcd_horizontal_flip, pcd_vertical_flip with list of augmented data with these specific parameters

## Extend and use custom pipelines

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

    ```python
    from mmdet.datasets import PIPELINES

    @PIPELINES.register_module()
    class MyTransform:

        def __call__(self, results):
            results['dummy'] = True
            return results
    ```

2. Import the new class.

    ```python
    from .my_pipeline import MyTransform
    ```

3. Use it in config files.

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
