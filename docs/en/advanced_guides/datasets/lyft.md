# Lyft Dataset

This page provides specific tutorials about the usage of MMDetection3D for Lyft dataset.

## Before Preparation

You can download Lyft 3D detection data [HERE](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data) and unzip all zip files.

Like the general way to prepare a dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.

The folder structure should be organized as follows before our processing.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (train_data)
│   │   │   ├── lidar (train_lidar)
│   │   │   ├── images (train_images)
│   │   │   ├── maps (train_maps)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (test_data)
│   │   │   ├── lidar (test_lidar)
│   │   │   ├── images (test_images)
│   │   │   ├── maps (test_maps)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
```

Here `v1.01-train` and `v1.01-test` contain the metafiles which are similar to those of nuScenes. `.txt` files contain the data split information.
Lyft does not have an official split for training and validation set, so we provide a split considering the number of objects from different categories in different scenes.
`sample_submission.csv` is the base file for submission on the Kaggle evaluation server.
Note that we follow the original folder names for clear organization. Please rename the raw folders as shown above.

## Dataset Preparation

The way to organize Lyft dataset is similar to nuScenes. We also generate the `.pkl` files which share almost the same structure.
Next, we will mainly focus on the difference between these two datasets. For a more detailed explanation of the info structure, please refer to [nuScenes tutorial](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/en/advanced_guides/datasets/nuscenes_det.md).

To prepare info files for Lyft, run the following commands:

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/dataset_converters/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

Note that the second command serves the purpose of fixing a corrupted lidar data file. Please refer to the discussion [here](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000) for more details.

The folder structure after processing should be as below.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (train_data)
│   │   │   ├── lidar (train_lidar)
│   │   │   ├── images (train_images)
│   │   │   ├── maps (train_maps)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (test_data)
│   │   │   ├── lidar (test_lidar)
│   │   │   ├── images (test_images)
│   │   │   ├── maps (test_maps)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
│   │   ├── lyft_infos_train.pkl
│   │   ├── lyft_infos_val.pkl
│   │   ├── lyft_infos_test.pkl
```

- `lyft_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `info_version`, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:
  - info\['sample_idx'\]: The index of this sample in the whole dataset.
  - info\['token'\]: Sample data token.
  - info\['timestamp'\]: Timestamp of the sample data.
  - info\['lidar_points'\]: A dict containing all the information related to the lidar points.
    - info\['lidar_points'\]\['lidar_path'\]: The filename of the lidar point cloud data.
    - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of point.
    - info\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)
    - info\['lidar_points'\]\['ego2global'\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
  - info\['lidar_sweeps'\]: A list contains sweeps information (The intermediate lidar frames without annotations).
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['data_path'\]: The lidar data path of i-th sweep.
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle in i-th sweep timestamp
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['ego2global'\]: The transformation matrix from the ego vehicle in i-th sweep timestamp to global coordinates. (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['lidar2sensor'\]: The transformation matrix from the keyframe lidar to the i-th frame lidar. (4x4 list)
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
    - info\['instances'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box in lidar coordinate system of the instance, in (x, y, z, l, w, h, yaw) order.
    - info\['instances'\]\[i\]\['bbox_label_3d'\]: A int starting from 0 indicates the label of instance, while the -1 indicates ignore class.
    - info\['instances'\]\[i\]\['bbox_3d_isvalid'\]: Whether each bounding box is valid. In general, we only take the 3D boxes that include at least one lidar or radar point as valid boxes.

Next, we will elaborate on the difference compared to nuScenes in terms of the details recorded in these info files.

- Without `lyft_database/xxxxx.bin`: This folder and `.bin` files are not extracted on the Lyft dataset due to the negligible effect of ground-truth sampling in the experiments.

- `lyft_infos_train.pkl`:

  - Without info\['instances'\]\[i\]\['velocity'\]: There is no velocity measurement on Lyft.
  - Without info\['instances'\]\[i\]\['num_lidar_pts'\] and info\['instances'\]\['num_radar_pts'\]

Here we only explain the data recorded in the training info files. The same applies to the validation set and test set (without instances).

Please refer to [lyft_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/lyft_converter.py) for more details about the structure of `lyft_infos_xxx.pkl`.

## Training pipeline

### LiDAR-Based Methods

A typical training pipeline of LiDAR-based 3D detection (including multi-modality methods) on Lyft is almost the same as nuScenes as below.

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
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

Similar to nuScenes, models on Lyft also need the `'LoadPointsFromMultiSweeps'` pipeline to load point clouds from consecutive frames.
In addition, considering the intensity of LiDAR points collected by Lyft is invalid, we also set the `use_dim` in `'LoadPointsFromMultiSweeps'` to `[0, 1, 2, 4]` by default,
where the first 3 dimensions refer to point coordinates, and the last refers to timestamp differences.

## Evaluation

An example to evaluate PointPillars with 8 GPUs with Lyft metrics is as follows:

```shell
bash ./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth 8
```

## Metrics

Lyft proposes a more strict metric for evaluating the predicted 3D bounding boxes.
The basic criteria to judge whether a predicted box is positive or not is the same as KITTI, i.e. the 3D Intersection over Union (IoU).
However, it adopts a way similar to COCO to compute the mean average precision (mAP) -- compute the average precision under different thresholds of 3D IoU from 0.5-0.95.
Actually, overlap more than 0.7 3D IoU is a quite strict criterion for 3D detection methods, so the overall performance seems a little low.
The imbalance of annotations for different categories is another important reason for the finally lower results compared to other datasets.
Please refer to its [official website](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview/evaluation) for more details about the definition of this metric.

We employ this official method for evaluation on Lyft. An example of printed evaluation results is as follows:

```
+mAPs@0.5:0.95------+--------------+
| class             | mAP@0.5:0.95 |
+-------------------+--------------+
| animal            | 0.0          |
| bicycle           | 0.099        |
| bus               | 0.177        |
| car               | 0.422        |
| emergency_vehicle | 0.0          |
| motorcycle        | 0.049        |
| other_vehicle     | 0.359        |
| pedestrian        | 0.066        |
| truck             | 0.176        |
| Overall           | 0.15         |
+-------------------+--------------+
```

## Testing and make a submission

An example to test PointPillars on Lyft with 8 GPUs and generate a submission to the leaderboard is as follows.

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py work_dirs/pp-lyft/latest.pth 8 --cfg-options test_evaluator.jsonfile_prefix=work_dirs/pp-lyft/results_challenge  test_evaluator.csv_savepath=results/pp-lyft/results_challenge.csv
```

After generating the `work_dirs/pp-lyft/results_challenge.csv`, you can submit it to the Kaggle evaluation server. Please refer to the [official website](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles) for more information.

We can also visualize the prediction results with our developed visualization tools. Please refer to the [visualization doc](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization) for more details.
