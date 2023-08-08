# NuScenes Dataset

This page provides specific tutorials about the usage of MMDetection3D for nuScenes dataset.

## Before Preparation

You can download nuScenes 3D detection `Full dataset (v1.0)` [HERE](https://www.nuscenes.org/download) and unzip all zip files.

If you want to implement 3D semantic segmentation task, you need to additionally download the `nuScenes-lidarseg` data annotation and place the extracted files in the nuScenes corresponding folder.

**Note**: `v1.0trainval(test)/categroy.json` in nuScenes-lidarseg will replace the original `v1.0trainval(test)/categroy.json` of the Full dataset (v1.0), but will not affect the 3D object detection task.

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.

The folder structure should be organized as follows before our processing.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## Dataset Preparation

We typically need to organize the useful data information with a `.pkl` file in a specific style.
To prepare these files for nuScenes, run the following command:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

The folder structure after processing should be as below.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

- `nuscenes_database/xxxxx.bin`: point cloud data included in each 3D bounding box of the training dataset
- `nuscenes_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `info_version`, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:
  - info\['sample_idx'\]: The index of this sample in the whole dataset.
  - info\['token'\]: Sample data token.
  - info\['timestamp'\]: Timestamp of the sample data.
  - info\['ego2global'\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
  - info\['lidar_points'\]: A dict containing all the information related to the lidar points.
    - info\['lidar_points'\]\['lidar_path'\]: The filename of the lidar point cloud data.
    - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of point.
    - info\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)
  - info\['lidar_sweeps'\]: A list contains sweeps information (The intermediate lidar frames without annotations)
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['data_path'\]: The lidar data path of i-th sweep.
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['ego2global'\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['lidar2sensor'\]: The transformation matrix from the main lidar sensor to the current sensor (for collecting the sweep data). (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['timestamp'\]: Timestamp of the sweep data.
    - info\['lidar_sweeps'\]\[i\]\['sample_data_token'\]: The sweep sample data token.
  - info\['images'\]: A dict contains six keys corresponding to each camera: `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`. Each dict contains all data information related to  corresponding camera.
    - info\['images'\]\['CAM_XXX'\]\['img_path'\]: The filename of the image.
    - info\['images'\]\['CAM_XXX'\]\['cam2img'\]: The transformation matrix recording the intrinsic parameters when projecting 3D points to each image plane. (3x3 list)
    - info\['images'\]\['CAM_XXX'\]\['sample_data_token'\]: Sample data token of image.
    - info\['images'\]\['CAM_XXX'\]\['timestamp'\]: Timestamp of the image.
    - info\['images'\]\['CAM_XXX'\]\['cam2ego'\]: The transformation matrix from this camera sensor to ego vehicle. (4x4 list)
    - info\['images'\]\['CAM_XXX'\]\['lidar2cam'\]: The transformation matrix from lidar sensor to this camera. (4x4 list)
  - info\['instances'\]: It is a list of dict. Each dict contains all annotation information of single instance. For the i-th instance:
    - info\['instances'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, w, h, yaw) order.
    - info\['instances'\]\[i\]\['bbox_label_3d'\]: A int indicate the label of instance and the -1 indicate ignore.
    - info\['instances'\]\[i\]\['velocity'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2.).
    - info\['instances'\]\[i\]\['num_lidar_pts'\]: Number of lidar points included in each 3D bounding box.
    - info\['instances'\]\[i\]\['num_radar_pts'\]: Number of radar points included in each 3D bounding box.
    - info\['instances'\]\[i\]\['bbox_3d_isvalid'\]: Whether each bounding box is valid. In general, we only take the 3D boxes that include at least one lidar or radar point as valid boxes.
  - info\['cam_instances'\]: It is a dict containing keys `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`. For vision-based 3D object detection task, we split 3D annotations of the whole scenes according to the camera they belong to. For the i-th instance:
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label'\]: Label of instance.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label_3d'\]: Label of instance.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox'\]: 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as \[x1, y1, x2, y2\].
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['center_2d'\]: Projected center location on the image, a list has shape (2,), .
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['depth'\]: The depth of projected center.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['velocity'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2,).
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['attr_label'\]: The attr label of instance. We maintain a default attribute collection and mapping for attribute classification.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, h, w, yaw) order.
  - info\['pts_semantic_mask_path'\]：The filename of the lidar point cloud semantic segmentation annotation.

Note:

1. The differences between `bbox_3d` in `instances` and that in `cam_instances`.
   Both `bbox_3d` have been converted to MMDet3D coordinate system, but `bboxes_3d` in `instances` is in LiDAR coordinate format and `bboxes_3d` in `cam_instances` is in Camera coordinate format. Mind the difference between them in 3D Box representation ('l, w, h' and 'l, h, w').

2. Here we only explain the data recorded in the training info files. The same applies to validation and testing set (the `.pkl` file of test set does not contains `instances` and `cam_instances`).

The core function to get `nuscenes_infos_xxx.pkl` is  [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/nuscenes_converter.py#L146).
Please refer to [nuscenes_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/nuscenes_converter.py) for more details.

## Training pipeline

### LiDAR-Based Methods

A typical training pipeline of LiDAR-based 3D detection (including multi-modality methods) on nuScenes is as below.

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
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
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

Compared to general cases, nuScenes has a specific `'LoadPointsFromMultiSweeps'` pipeline to load point clouds from consecutive frames. This is a common practice used in this setting.
Please refer to the nuScenes [original paper](https://arxiv.org/abs/1903.11027) for more details.
The default `use_dim` in `'LoadPointsFromMultiSweeps'` is `[0, 1, 2, 4]`, where the first 3 dimensions refer to point coordinates and the last refers to timestamp differences.
Intensity is not used by default due to its yielded noise when concatenating the points from different frames.

### Vision-Based Methods

#### Monocular-based

In the NuScenes dataset, for multi-view images, this paradigm usually involves detecting and outputting 3D object detection results separately for each image, and then obtaining the final detection results through post-processing (such as NMS). Essentially, it directly extends monocular 3D detection to multi-view settings. A typical training pipeline of image-based monocular 3D detection on nuScenes is as below.

```python
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
```

It follows the general pipeline of 2D detection while differs in some details:

- It uses monocular pipelines to load images, which includes additional required information like camera intrinsics.
- It needs to load 3D annotations.
- Some data augmentation techniques need to be adjusted, such as `RandomFlip3D`.
  Currently we do not support more augmentation methods, because how to transfer and apply other techniques is still under explored.

#### BEV-based

BEV, Bird's-Eye-View, is another popular 3D detection paradigm. It directly takes multi-view images to perform 3D detection, for nuScenes, they are `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT` and `CAM_BACK_RIGHT`. A basic training pipeline of bev-based 3D detection on nuScenes is as below.

```python
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
train_transforms = [
    dict(type='PhotoMetricDistortion3D'),
    dict(
        type='RandomResize3D',
        scale=(1600, 900),
        ratio_range=(1., 1.),
        keep_ratio=True)
]
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=6, ),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    # optional, data augmentation
    dict(type='MultiViewWrapper', transforms=train_transforms),
    # optional, filter object within specific point cloud range
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # optional, filter object of specific classes
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

To load multiple view of images, a little modification should be made to the dataset.

```python
data_prefix = dict(
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
)
train_dataloader = dict(
    batch_size=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type="NuScenesDataset",
        data_root="./data/nuScenes",
        ann_file="nuscenes_infos_train.pkl",
        data_prefix=data_prefix,
        modality=dict(use_camera=True, use_lidar=False, ),
        pipeline=train_pipeline,
        test_mode=False, )
)
```

## Evaluation

An example to evaluate PointPillars with 8 GPUs with nuScenes metrics is as follows.

```shell
bash ./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth 8
```

## Metrics

NuScenes proposes a comprehensive metric, namely nuScenes detection score (NDS), to evaluate different methods and set up the benchmark.
It consists of mean Average Precision (mAP), Average Translation Error (ATE), Average Scale Error (ASE), Average Orientation Error (AOE), Average Velocity Error (AVE) and Average Attribute Error (AAE).
Please refer to its [official website](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) for more details.

We also adopt this approach for evaluation on nuScenes. An example of printed evaluation results is as follows:

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.503   0.577   0.152   0.111   2.096   0.136
truck   0.223   0.857   0.224   0.220   1.389   0.179
bus     0.294   0.855   0.204   0.190   2.689   0.283
trailer 0.081   1.094   0.243   0.553   0.742   0.167
construction_vehicle    0.058   1.017   0.450   1.019   0.137   0.341
pedestrian      0.392   0.687   0.284   0.694   0.876   0.158
motorcycle      0.317   0.737   0.265   0.580   2.033   0.104
bicycle 0.308   0.704   0.299   0.892   0.683   0.010
traffic_cone    0.555   0.486   0.309   nan     nan     nan
barrier 0.466   0.581   0.269   0.169   nan     nan
```

## Testing and make a submission

An example to test PointPillars on nuScenes with 8 GPUs and generate a submission to the leaderboard is as follows.

You should modify the `jsonfile_prefix` in the `test_evaluator` of corresponding configuration. For example, adding `test_evaluator = dict(type='NuScenesMetric', jsonfile_prefix='work_dirs/pp-nus/results_eval.json')` or using `--cfg-options "test_evaluator.jsonfile_prefix=work_dirs/pp-nus/results_eval.json)` after the test command.

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py work_dirs/pp-nus/latest.pth 8 --cfg-options 'test_evaluator.jsonfile_prefix=work_dirs/pp-nus/results_eval'
```

Note that the testing info should be changed to that for testing set instead of validation set [here](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/nus-3d.py#L132).

After generating the `work_dirs/pp-nus/results_eval.json`, you can compress it and submit it to nuScenes benchmark. Please refer to the [nuScenes official website](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) for more information.

We can also visualize the prediction results with our developed visualization tools. Please refer to the [visualization doc](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization) for more details.

## Notes

### Transformation between `NuScenesBox` and our `CameraInstanceBoxes`.

In general, the main difference of `NuScenesBox` and our `CameraInstanceBoxes` is mainly reflected in the yaw definition. `NuScenesBox` defines the rotation with a quaternion or three Euler angles while ours only defines one yaw angle due to the practical scenario. It requires us to add some additional rotations manually in the pre-processing and post-processing, such as [here](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L673).

In addition, please note that the definition of corners and locations are detached in the `NuScenesBox`. For example, in monocular 3D detection, the definition of the box location is in its camera coordinate (see its official [illustration](https://www.nuscenes.org/nuscenes#data-collection) for car setup), which is consistent with [ours](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py). In contrast, its corners are defined with the [convention](https://github.com/nutonomy/nuscenes-devkit/blob/02e9200218977193a1058dd7234f935834378319/python-sdk/nuscenes/utils/data_classes.py#L527) "x points forward, y to the left, z up". It results in different philosophy of dimension and rotation definitions from our `CameraInstanceBoxes`. An example to remove similar hacks is PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744). The same problem also exists in the LiDAR system. To deal with them, we typically add some transformation in the pre-processing and post-processing to guarantee the box will be in our coordinate system during the entire training and inference procedure.
