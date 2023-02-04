# Customize Datasets

In this note, you will know how to train and test predefined models with customized datasets.

The basic steps are as below:

1. Prepare data
2. Prepare a config
3. Train, test and inference models on the customized dataset

## Data Preparation

The ideal situation is that we can reorganize the customized raw data and convert the annotation format into KITTI style. However, considering some calibration files and 3D annotations in KITTI format are difficult to obtain for customized datasets, we introduce the basic data format in the doc.

### Basic Data Format

#### Point cloud Format

Currently, we only support `.bin` format point cloud for training and inference. Before training on your own datasets, you need to convert your point cloud files with other formats to `.bin` files. The common point cloud data formats include `.pcd` and `.las`, we list some open-source tools for reference.

1. Convert `.pcd` to `.bin`: https://github.com/DanielPollithy/pypcd

- You can install `pypcd` with the following command:

  ```bash
  pip install git+https://github.com/DanielPollithy/pypcd.git
  ```

- You can use the following script to read the `.pcd` file and convert it to `.bin` format for saving:

  ```python
  import numpy as np
  from pypcd import pypcd

  pcd_data = pypcd.PointCloud.from_path('point_cloud_data.pcd')
  points = np.zeros([pcd_data.width, 4], dtype=np.float32)
  points[:, 0] = pcd_data.pc_data['x'].copy()
  points[:, 1] = pcd_data.pc_data['y'].copy()
  points[:, 2] = pcd_data.pc_data['z'].copy()
  points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
  with open('point_cloud_data.bin', 'wb') as f:
      f.write(points.tobytes())
  ```

2. Convert `.las` to `.bin`: The common conversion path is `.las -> .pcd -> .bin`, and the conversion path `.las -> .pcd` can be achieved through [this tool](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor).

#### Label Format

The most basic information: 3D bounding box and category label of each scene need to be contained in the `.txt` annotation file. Each line represents a 3D box in a certain scene as follow:

```
# format: [x, y, z, dx, dy, dz, yaw, category_name]
1.23 1.42 0.23 3.96 1.65 1.55 1.56 Car
3.51 2.15 0.42 1.05 0.87 1.86 1.23 Pedestrian
...
```

**Note**: Currently we only support KITTI Metric evaluation for customized datasets evaluation.

The 3D Box should be stored in unified 3D coordinates.

#### Calibration Format

For the point cloud data collected by each LiDAR, they are usually fused and converted to a certain LiDAR coordinate. So typically the calibration information file should contain the intrinsic matrix of each camera and the transformation extrinsic matrix from the LiDAR to each camera in `.txt` calibration file, while `Px` represents the intrinsic matrix of `camera_x` and `lidar2camx` represents the transformation extrinsic matrix from the `lidar` to `camera_x`.

```
P0
P1
P2
P3
P4
...
lidar2cam0
lidar2cam1
lidar2cam2
lidar2cam3
lidar2cam4
...
```

### Raw Data Structure

#### LiDAR-Based 3D Detection

The raw data for LiDAR-based 3D object detection are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation set, `points` includes point cloud data which are supposed to be stored in `.bin` format and `labels` includes label files for 3D detection.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### Vision-Based 3D Detection

The raw data for vision-based 3D object detection are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation set, `images` contains the images from different cameras, for example, images from `camera_x` need to be placed in `images/images_x`, `calibs` contains calibration information files which store the camera intrinsic matrix of each camera, and `labels` includes label files for 3D detection.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### Multi-Modality 3D Detection

The raw data for multi-modality 3D object detection are typically organized as follows. Different from vision-based 3D object detection, calibration information files in `calibs` store the camera intrinsic matrix of each camera and extrinsic matrix.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### LiDAR-Based 3D Semantic Segmentation

The raw data for LiDAR-based 3D semantic segmentation are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation set, `points` includes point cloud data, and `semantic_mask` includes point-level label.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── semantic_mask
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
```

### Data Converter

Once you prepared the raw data following our instruction, you can directly use the following command to generate training/validation information files.

```bash
python tools/create_data.py custom --root-path ./data/custom --out-dir ./data/custom --extra-tag custom
```

## An example of customized dataset

Once we finish data preparation, we can create a new dataset in `mmdet3d/datasets/my_dataset.py` to load the data.

```python
import mmengine

from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MyDataset(Det3DDataset):

    # replace with all the classes in customized pkl info file
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car')
    }

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
```

After the data pre-processing, there are two steps for users to train the customized new dataset:

1. Modify the config file for using the customized dataset.
2. Check the annotations of the customized dataset.

Here we take training PointPillars on customized dataset as an example:

### Prepare a config

Here we demonstrate a config sample for pure point cloud training.

#### Prepare dataset config

In `configs/_base_/datasets/custom.py`:

```python
# dataset settings
dataset_type = 'MyDataset'
data_root = 'data/custom/'
class_names = ['Pedestrian', 'Cyclist', 'Car']  # replace with your dataset class
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # adjust according to your dataset
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),  # replace with the actual dimension used in training and inference
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
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
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
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
            ann_file='custom_infos_train.pkl',  # specify your training pkl info
            data_prefix=dict(pts='points'),
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
        data_prefix=dict(pts='points'),
        ann_file='custom_infos_val.pkl',  # specify your validation pkl info
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
```

#### Prepare model config

For voxel-based detectors such as SECOND, PointPillars and CenterPoint, the point cloud range and voxel size should be adjusted according to your dataset.
Theoretically, `voxel_size` is linked to the setting of `point_cloud_range`. Setting a smaller `voxel_size` will increase the voxel num and the corresponding memory consumption. In addition, the following issues need to be noted:

If the `point_cloud_range` and `voxel_size` are set to be `[0, -40, -3, 70.4, 40, 1]` and `[0.05, 0.05, 0.1]` respectively, then the shape of intermediate feature map should be `[(1-(-3))/0.1+1, (40-(-40))/0.05, (70.4-0)/0.05]=[41, 1600, 1408]`. When changing `point_cloud_range`, remember to change the shape of intermediate feature map in `middle_encoder` according to the `voxel_size`.

Regarding the setting of `anchor_range`, it is generally adjusted according to dataset. Note that `z` value needs to be adjusted accordingly to the position of the point cloud, please refer to this [issue](https://github.com/open-mmlab/mmdetection3d/issues/986).

Regarding the setting of `anchor_size`, it is usually necessary to count the average length, width and height of objects in the entire training dataset as `anchor_size` to obtain the best results.

In `configs/_base_/models/pointpillars_hv_secfpn_custom.py`:

```python
voxel_size = [0.16, 0.16, 4]  # adjust according to your dataset
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # adjust according to your dataset
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    # the `output_shape` should be adjusted according to `point_cloud_range`
    # and `voxel_size`
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
        # adjust the `ranges` and `sizes` according to your dataset
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
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
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
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

#### Prepare overall config

We combine all the configs above in `configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py`:

```python
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_custom.py',
    '../_base_/datasets/custom.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
```

#### Visualize your dataset (optional)

To validate whether your prepared data and config are correct, it's highly recommended to use `tools/misc/browse_dataset.py` script
to visualize your dataset and annotations before training and validation. Please refer to [visualization doc](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/visualization.html) for more details.

## Evaluation

Once the data and config have been prepared, you can directly run the training/testing script following our doc.

**Note**: We only provide an implementation for KITTI style evaluation for the customized dataset. It should be included in the dataset config:

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
```
