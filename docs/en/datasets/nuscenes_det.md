# NuScenes Dataset for 3D Object Detection

This page provides specific tutorials about the usage of MMDetection3D for nuScenes dataset.

## Before Preparation

You can download nuScenes 3D detection data [HERE](https://www.nuscenes.org/download) and unzip all zip files.

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
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## Dataset Preparation

We typically need to organize the useful data information with a .pkl or .json file in a specific style, e.g., coco-style for organizing images and their annotations.
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
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json
```

Here, .pkl files are generally used for methods involving point clouds and coco-style .json files are more suitable for image-based methods, such as image-based 2D and 3D detection.
Next, we will elaborate on the details recorded in these info files.

- `nuscenes_database/xxxxx.bin`: point cloud data included in each 3D bounding box of the training dataset
- `nuscenes_infos_train.pkl`: training dataset info, each frame info has two keys: `metadata` and `infos`.
  `metadata` contains the basic information for the dataset itself, such as `{'version': 'v1.0-trainval'}`, while `infos` contains the detailed information as follows:
  - info\['lidar_path'\]: The file path of the lidar point cloud data.
  - info\['token'\]: Sample data token.
  - info\['sweeps'\]: Sweeps information (`sweeps` in the nuScenes refer to the intermediate frames without annotations, while `samples` refer to those key frames with annotations).
    - info\['sweeps'\]\[i\]\['data_path'\]: The data path of i-th sweep.
    - info\['sweeps'\]\[i\]\['type'\]: The sweep data type, e.g., `'lidar'`.
    - info\['sweeps'\]\[i\]\['sample_data_token'\]: The sweep sample data token.
    - info\['sweeps'\]\[i\]\['sensor2ego_translation'\]: The translation from the current sensor (for collecting the sweep data) to ego vehicle. (1x3 list)
    - info\['sweeps'\]\[i\]\['sensor2ego_rotation'\]: The rotation from the current sensor (for collecting the sweep data) to ego vehicle. (1x4 list in the quaternion format)
    - info\['sweeps'\]\[i\]\['ego2global_translation'\]: The translation from the ego vehicle to global coordinates. (1x3 list)
    - info\['sweeps'\]\[i\]\['ego2global_rotation'\]: The rotation from the ego vehicle to global coordinates. (1x4 list in the quaternion format)
    - info\['sweeps'\]\[i\]\['timestamp'\]: Timestamp of the sweep data.
    - info\['sweeps'\]\[i\]\['sensor2lidar_translation'\]: The translation from the current sensor (for collecting the sweep data) to lidar. (1x3 list)
    - info\['sweeps'\]\[i\]\['sensor2lidar_rotation'\]: The rotation from the current sensor (for collecting the sweep data) to lidar. (1x4 list in the quaternion format)
  - info\['cams'\]: Cameras calibration information. It contains six keys corresponding to each camera: `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`.
    Each dictionary contains detailed information following the above way for each sweep data (has the same keys for each information as above). In addition, each camera has a key `'cam_intrinsic'` for recording the intrinsic parameters when projecting 3D points to each image plane.
  - info\['lidar2ego_translation'\]: The translation from lidar to ego vehicle. (1x3 list)
  - info\['lidar2ego_rotation'\]: The rotation from lidar to ego vehicle. (1x4 list in the quaternion format)
  - info\['ego2global_translation'\]: The translation from the ego vehicle to global coordinates. (1x3 list)
  - info\['ego2global_rotation'\]: The rotation from the ego vehicle to global coordinates. (1x4 list in the quaternion format)
  - info\['timestamp'\]: Timestamp of the sample data.
  - info\['gt_boxes'\]: 7-DoF annotations of 3D bounding boxes, an Nx7 array.
  - info\['gt_names'\]: Categories of 3D bounding boxes, an 1xN array.
  - info\['gt_velocity'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), an Nx2 array.
  - info\['num_lidar_pts'\]: Number of lidar points included in each 3D bounding box.
  - info\['num_radar_pts'\]: Number of radar points included in each 3D bounding box.
  - info\['valid_flag'\]: Whether each bounding box is valid. In general, we only take the 3D boxes that include at least one lidar or radar point as valid boxes.
- `nuscenes_infos_train_mono3d.coco.json`: training dataset coco-style info. This file organizes image-based data into three categories (keys): `'categories'`, `'images'`, `'annotations'`.
  - info\['categories'\]: A list containing all the category names. Each element follows the dictionary format and consists of two keys: `'id'` and `'name'`.
  - info\['images'\]: A list containing all the image info.
    - info\['images'\]\[i\]\['file_name'\]: The file name of the i-th image.
    - info\['images'\]\[i\]\['id'\]: Sample data token of the i-th image.
    - info\['images'\]\[i\]\['token'\]: Sample token corresponding to this frame.
    - info\['images'\]\[i\]\['cam2ego_rotation'\]: The rotation from the camera to ego vehicle. (1x4 list in the quaternion format)
    - info\['images'\]\[i\]\['cam2ego_translation'\]: The translation from the camera to ego vehicle. (1x3 list)
    - info\['images'\]\[i\]\['ego2global_rotation''\]: The rotation from the ego vehicle to global coordinates. (1x4 list in the quaternion format)
    - info\['images'\]\[i\]\['ego2global_translation'\]: The translation from the ego vehicle to global coordinates. (1x3 list)
    - info\['images'\]\[i\]\['cam_intrinsic'\]: Camera intrinsic matrix. (3x3 list)
    - info\['images'\]\[i\]\['width'\]: Image width, 1600 by default in nuScenes.
    - info\['images'\]\[i\]\['height'\]: Image height, 900 by default in nuScenes.
  - info\['annotations'\]: A list containing all the annotation info.
    - info\['annotations'\]\[i\]\['file_name'\]: The file name of the corresponding image.
    - info\['annotations'\]\[i\]\['image_id'\]: The image id (token) of the corresponding image.
    - info\['annotations'\]\[i\]\['area'\]: Area of the 2D bounding box.
    - info\['annotations'\]\[i\]\['category_name'\]: Category name.
    - info\['annotations'\]\[i\]\['category_id'\]: Category id.
    - info\['annotations'\]\[i\]\['bbox'\]: 2D bounding box annotation (exterior rectangle of the projected 3D box), 1x4 list following \[x1, y1, x2-x1, y2-y1\].
      x1/y1 are minimum coordinates along horizontal/vertical direction of the image.
    - info\['annotations'\]\[i\]\['iscrowd'\]: Whether the region is crowded. Defaults to 0.
    - info\['annotations'\]\[i\]\['bbox_cam3d'\]: 3D bounding box (gravity) center location (3), size (3), (global) yaw angle (1), 1x7 list.
    - info\['annotations'\]\[i\]\['velo_cam3d'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), an Nx2 array.
    - info\['annotations'\]\[i\]\['center2d'\]: Projected 3D-center containing 2.5D information: projected center location on the image (2) and depth (1), 1x3 list.
    - info\['annotations'\]\[i\]\['attribute_name'\]: Attribute name.
    - info\['annotations'\]\[i\]\['attribute_id'\]: Attribute id.
      We maintain a default attribute collection and mapping for attribute classification.
      Please refer to [here](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L53) for more details.
    - info\['annotations'\]\[i\]\['id'\]: Annotation id. Defaults to `i`.

Here we only explain the data recorded in the training info files. The same applies to validation and testing set.

The core function to get `nuscenes_infos_xxx.pkl` and `nuscenes_infos_xxx_mono3d.coco.json` are [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py#L143) and [get_2d_boxes](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py#L397), respectively.
Please refer to [nuscenes_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py) for more details.

## Training pipeline

### LiDAR-Based Methods

A typical training pipeline of LiDAR-based 3D detection (including multi-modality methods) on nuScenes is as below.

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
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
```

Compared to general cases, nuScenes has a specific `'LoadPointsFromMultiSweeps'` pipeline to load point clouds from consecutive frames. This is a common practice used in this setting.
Please refer to the nuScenes [original paper](https://arxiv.org/abs/1903.11027) for more details.
The default `use_dim` in `'LoadPointsFromMultiSweeps'` is `[0, 1, 2, 4]`, where the first 3 dimensions refer to point coordinates and the last refers to timestamp differences.
Intensity is not used by default due to its yielded noise when concatenating the points from different frames.

### Vision-Based Methods

A typical training pipeline of image-based 3D detection on nuScenes is as below.

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
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
```

It follows the general pipeline of 2D detection while differs in some details:

- It uses monocular pipelines to load images, which includes additional required information like camera intrinsics.
- It needs to load 3D annotations.
- Some data augmentation techniques need to be adjusted, such as `RandomFlip3D`.
  Currently we do not support more augmentation methods, because how to transfer and apply other techniques is still under explored.

## Evaluation

An example to evaluate PointPillars with 8 GPUs with nuScenes metrics is as follows.

```shell
bash ./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth 8 --eval bbox
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

```shell
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py work_dirs/pp-nus/latest.pth 8 --out work_dirs/pp-nus/results_eval.pkl --format-only --eval-options 'jsonfile_prefix=work_dirs/pp-nus/results_eval'
```

Note that the testing info should be changed to that for testing set instead of validation set [here](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/nus-3d.py#L132).

After generating the `work_dirs/pp-nus/results_eval.json`, you can compress it and submit it to nuScenes benchmark. Please refer to the [nuScenes official website](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) for more information.

We can also visualize the prediction results with our developed visualization tools. Please refer to the [visualization doc](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization) for more details.

## Notes

### Transformation between `NuScenesBox` and our `CameraInstanceBoxes`.

In general, the main difference of `NuScenesBox` and our `CameraInstanceBoxes` is mainly reflected in the yaw definition. `NuScenesBox` defines the rotation with a quaternion or three Euler angles while ours only defines one yaw angle due to the practical scenario. It requires us to add some additional rotations manually in the pre-processing and post-processing, such as [here](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L673).

In addition, please note that the definition of corners and locations are detached in the `NuScenesBox`. For example, in monocular 3D detection, the definition of the box location is in its camera coordinate (see its official [illustration](https://www.nuscenes.org/nuscenes#data-collection) for car setup), which is consistent with [ours](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py). In contrast, its corners are defined with the [convention](https://github.com/nutonomy/nuscenes-devkit/blob/02e9200218977193a1058dd7234f935834378319/python-sdk/nuscenes/utils/data_classes.py#L527) "x points forward, y to the left, z up". It results in different philosophy of dimension and rotation definitions from our `CameraInstanceBoxes`. An example to remove similar hacks is PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744). The same problem also exists in the LiDAR system. To deal with them, we typically add some transformation in the pre-processing and post-processing to guarantee the box will be in our coordinate system during the entire training and inference procedure.
