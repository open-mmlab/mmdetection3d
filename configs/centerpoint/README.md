# Center-based 3D Object Detection and Tracking

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
```
@article{yin2020center,
  title={Center-based 3d object detection and tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2006.11275},
  year={2020}
}
```

## Usage

### Test time augmentation

We have supported double-flip and scale augmentation during test time. To use test time augmentation by modifying pipeline and test_cfg in the config.
For example, we change `centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py` to the following.
```python
_base_ = './centerpoint_0075voxel_second_secfpn_circlenms' \
         '_4x8_cyclic_20e_nus.py'

test_cfg = dict(
    pts=dict(
        use_rotate_nms=True,
        max_num=83))

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
        ],
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True)
]

data = dict(
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))

```

## Results

### CenterPoint

|Backbone|  Voxel type (voxel size)   |Dcn|Circular nms| Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: |:-----: |:-----: | :------: | :------------: | :----: |:----: | :------: |:------: |
|[SECFPN](./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✗||||||
|[SECFPN](./centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✓||||||
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✗||||||
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✓||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✗||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✓||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✗||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✓||||||
|[SECFPN](./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✗||||||
|[SECFPN](./centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✓|||48.72|59.40||
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✗||||||
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✓||||||
