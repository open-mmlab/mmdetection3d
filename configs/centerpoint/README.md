# Center-based 3D Object Detection and Tracking

## Introduction

<!-- [ALGORITHM] -->

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
@article{yin2021center,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal={CVPR},
  year={2021},
}
```

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

## Results

### CenterPoint

|Backbone|  Voxel type (voxel size)   |Dcn|Circular nms| Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: |:-----: |:-----: | :------: | :------------: | :----: |:----: | :------: |:------: |
|[SECFPN](./centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✓|5.3| |56.44|64.63|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210815_085857-9ba7f3a5.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210815_085857.log.json)|
|above w/o circle nms|voxel (0.1)|✗|✗|5.3| |56.03|64.62|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_20210816_072453-d2bbdc71.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_20210816_072453.log.json)|
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✓|5.5| |56.00|64.44|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210814_060754-c9d535d2.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210814_060754.log.json)|
|above w/o circle nms|voxel (0.1)|✓|✗|5.5| |56.45|64.48|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_023812-fbf8dd33.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_023812.log.json)|
|[SECFPN](./centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✓|8.2| |56.62|64.79|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210814_113418-76ae0cf0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210814_113418.log.json)|
|above w/o circle nms|voxel (0.075)|✗|✗|8.2| |57.39|65.45|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus_20210814_132346-340fe7d5.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus_20210814_132346.log.json)|
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✓|8.5| |56.35|65.21|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135-1782af3e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135.log.json)|
|above w/o circle nms|voxel (0.075)|✓|✗|8.7| |57.13|65.54|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135-1782af3e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135.log.json)|
|above w/ double flip|voxel (0.075)|✓|✗|8.7| |59.64|67.46|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_flip-tta_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_flip-tta_20e_nus_20210822_021103-48ee1a9a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_flip-tta_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_flip-tta_20e_nus_20210822_021103.log.json)|
|above w/ scale tta|voxel (0.075)|✓|✗|8.7| |60.23|67.34|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_tta_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_tta_20e_nus_20210824_090810-3b7add71.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_tta_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_tta_20e_nus_20210824_090810.log.json)|
|above w/ circle nms w/o scale tta|voxel (0.075)|✓|✗|8.7| |59.51|67.12|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_nus_20210814_065655-4eb563f6.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_nus_20210814_065655.log.json)|
|[SECFPN](./centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✓|4.4| |48.33|59.12|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624-0f3299c0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624.log.json)|
|above w/o circle nms|pillar (0.2)|✗|✗|4.4| |48.15|58.83|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_20210814_162937-be654cec.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_20210814_162937.log.json)|
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✗|4.6| |48.57|59.50|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_202702-f03ab9e4.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_202702.log.json)|
|above w/ circle nms|pillar (0.2)|✓|✓|4.6| |48.63|59.27|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210815_005003-7561138d.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210815_005003.log.json)|
