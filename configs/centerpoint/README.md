# Center-based 3D Object Detection and Tracking

> [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)

<!-- [ALGORITHM] -->

## Abstract

Three-dimensional objects are commonly represented as 3D boxes in a point-cloud. This representation mimics the well-studied image-based 2D bounding-box detection but comes with additional challenges. Objects in a 3D world do not follow any particular orientation, and box-based detectors have difficulties enumerating all orientations or fitting an axis-aligned bounding box to rotated objects. In this paper, we instead propose to represent, detect, and track 3D objects as points. Our framework, CenterPoint, first detects centers of objects using a keypoint detector and regresses to other attributes, including 3D size, 3D orientation, and velocity. In a second stage, it refines these estimates using additional point features on the object. In CenterPoint, 3D object tracking simplifies to greedy closest-point matching. The resulting detection and tracking algorithm is simple, efficient, and effective. CenterPoint achieved state-of-the-art performance on the nuScenes benchmark for both 3D detection and tracking, with 65.5 NDS and 63.8 AMOTA for a single model. On the Waymo Open Dataset, CenterPoint outperforms all previous single model method by a large margin and ranks first among all Lidar-only submissions.

<div align=center>
<img src="https://user-images.githubusercontent.com/30491025/143854976-11af75ae-e828-43ad-835d-ac1146f99925.png" width="800"/>
</div>

## Introduction

We implement CenterPoint and provide the result and checkpoints on nuScenes dataset.

We follow the below style to name config files. Contributors are advised to follow the same style.
`{xxx}` is required field and `[yyy]` is optional.

`{model}`: model type like `centerpoint`.

`{model setting}`: voxel size and voxel type like `01voxel`, `02pillar`.

`{backbone}`: backbone type like `second`.

`{neck}`: neck type like `secfpn`.

`[dcn]`: Whether to use deformable convolution.

`[circle]`: Whether to use circular nms.

`[batch_per_gpu x gpu]`: GPUs and samples per GPU, 4x8 is used by default.

`{schedule}`: training schedule, options are 1x, 2x, 20e, etc. 1x and 2x means 12 epochs and 24 epochs respectively. 20e is adopted in cascade models, which denotes 20 epochs. For 1x/2x, initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs. For 20e, initial learning rate decays by a factor of 10 at the 16th and 19th epochs.

`{dataset}`: dataset like nus-3d, kitti-3d, lyft-3d, scannet-3d, sunrgbd-3d. We also indicate the number of classes we are using if there exist multiple settings, e.g., kitti-3d-3class and kitti-3d-car means training on KITTI dataset with 3 classes and single class, respectively.

## Usage

### Test time augmentation

We have supported double-flip and scale augmentation during test time. To use test time augmentation, users need to modify the
`test_pipeline` and `test_cfg` in the config.
For example, we change `centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py` to the following.

```python
_base_ = './centerpoint_0075voxel_second_secfpn_circlenms' \
         '_4x8_cyclic_20e_nus.py'

model = dict(
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            max_num=83)))

point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
file_client_args = dict(backend='disk')
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=[0.95, 1.0, 1.05],
        flip=True,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))

```

## Results and models

### CenterPoint

|                                      Backbone                                       | Voxel type (voxel size) | Dcn | Circular nms | Mem (GB) | Inf time (fps) |  mAP  |  NDS  |                                                                                                                                                                                                                                                  Download                                                                                                                                                                                                                                                  |
| :---------------------------------------------------------------------------------: | :---------------------: | :-: | :----------: | :------: | :------------: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    [SECFPN](./centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)    |       voxel (0.1)       |  ✗  |      ✓       |   4.9    |                | 56.19 | 64.43 |             [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210815_085857-9ba7f3a5.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210815_085857.log.json)             |
|                                above w/o circle nms                                 |       voxel (0.1)       |  ✗  |      ✗       |          |                | 56.56 | 64.46 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|  [SECFPN](./centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)  |       voxel (0.1)       |  ✓  |      ✓       |   5.2    |                | 56.34 | 64.81 |     [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210814_060754-c9d535d2.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210814_060754.log.json)     |
|                                above w/o circle nms                                 |       voxel (0.1)       |  ✓  |      ✗       |          |                | 56.60 | 64.90 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|   [SECFPN](./centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)   |      voxel (0.075)      |  ✗  |      ✓       |   7.8    |                | 57.34 | 65.23 |         [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210814_113418-76ae0cf0.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210814_113418.log.json)         |
|                                above w/o circle nms                                 |      voxel (0.075)      |  ✗  |      ✗       |          |                | 57.63 | 65.39 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py) |      voxel (0.075)      |  ✓  |      ✓       |   8.5    |                | 57.27 | 65.58 | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135-1782af3e.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135.log.json) |
|                                above w/o circle nms                                 |      voxel (0.075)      |  ✓  |      ✗       |          |                | 57.43 | 65.63 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|                                above w/ double flip                                 |      voxel (0.075)      |  ✓  |      ✗       |          |                | 59.73 | 67.39 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|                                 above w/ scale tta                                  |      voxel (0.075)      |  ✓  |      ✗       |          |                | 60.43 | 67.65 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|                          above w/ circle nms w/o scale tta                          |      voxel (0.075)      |  ✓  |      ✗       |          |                | 59.52 | 67.24 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|   [SECFPN](./centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)    |      pillar (0.2)       |  ✗  |      ✓       |   4.4    |                | 49.07 | 59.66 |           [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624-0f3299c0.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624.log.json)           |
|                                above w/o circle nms                                 |      pillar (0.2)       |  ✗  |      ✗       |          |                | 49.12 | 59.66 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|      [SECFPN](./centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py)       |      pillar (0.2)       |  ✓  |      ✗       |   4.6    |                | 48.8  | 59.67 |                       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_202702-f03ab9e4.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_202702.log.json)                       |
|                                 above w/ circle nms                                 |      pillar (0.2)       |  ✓  |      ✓       |          |                | 48.79 | 59.65 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

## Citation

```latex
@article{yin2021center,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal={CVPR},
  year={2021},
}
```
