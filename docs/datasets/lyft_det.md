# Lyft Dataset for 3D Object Detection

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

The way to organize Lyft dataset is similar to nuScenes. We also generate the .pkl and .json files which share almost the same structure.
Next, we will mainly focus on the difference between these two datasets. For a more detailed explanation of the info structure, please refer to [nuScenes tutorial](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/datasets/nuscenes_det.md).

To prepare info files for Lyft, run the following commands:

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/data_converter/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
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
│   │   ├── lyft_infos_train_mono3d.coco.json
│   │   ├── lyft_infos_val_mono3d.coco.json
│   │   ├── lyft_infos_test_mono3d.coco.json
```

Here, .pkl files are generally used for methods involving point clouds, and coco-style .json files are more suitable for image-based methods, such as image-based 2D and 3D detection.
Different from nuScenes, we only support using the json files for 2D detection experiments. Image-based 3D detection may be further supported in the future.

Next, we will elaborate on the difference compared to nuScenes in terms of the details recorded in these info files.

- without `lyft_database/xxxxx.bin`: This folder and `.bin` files are not extracted on the Lyft dataset due to the negligible effect of ground-truth sampling in the experiments.
- `lyft_infos_train.pkl`: training dataset infos, each frame info has two keys: `metadata` and `infos`.
`metadata` contains the basic information for the dataset itself, such as `{'version': 'v1.01-train'}`, while `infos` contains the detailed information the same as nuScenes except for the following details:
    - info['sweeps']: Sweeps information.
        - info['sweeps'][i]['type']: The sweep data type, e.g., `'lidar'`.
          Lyft has different LiDAR settings for some samples, but we always take only the points collected by the top LiDAR for the consistency of data distribution.
    - info['gt_names']: There are 9 categories on the Lyft dataset, and the imbalance of annotations for different categories is even more significant than nuScenes.
    - without info['gt_velocity']: There is no velocity measurement on Lyft.
    - info['num_lidar_pts']: Set to -1 by default.
    - info['num_radar_pts']: Set to 0 by default.
    - without info['valid_flag']: This flag does recorded due to invalid `num_lidar_pts` and `num_radar_pts`.
- `nuscenes_infos_train_mono3d.coco.json`: training dataset coco-style info. This file only contains 2D information, without the information required by 3D detection, such as camera intrinsics.
    - info['images']: A list containing all the image info.
        - only containing `'file_name'`, `'id'`, `'width'`, `'height'`.
    - info['annotations']: A list containing all the annotation info.
        - only containing `'file_name'`, `'image_id'`, `'area'`, `'category_name'`, `'category_id'`, `'bbox'`, `'is_crowd'`, `'segmentation'`, `'id'`, where `'is_crowd'`, `'segmentation'` are set to `0` and `[]` by default.
        There is no attribute annotation on Lyft.

Here we only explain the data recorded in the training info files. The same applies to the testing set.

The core function to get `lyft_infos_xxx.pkl` is [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/lyft_converter.py#L91).
Please refer to [lyft_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/lyft_converter.py) for more details.

## Training pipeline

### LiDAR-Based Methods

A typical training pipeline of LiDAR-based 3D detection (including multi-modality methods) on Lyft is almost the same as nuScenes as below.

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
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

Similar to nuScenes, models on Lyft also need the `'LoadPointsFromMultiSweeps'` pipeline to load point clouds from consecutive frames.
In addition, considering the intensity of LiDAR points collected by Lyft is invalid, we also set the `use_dim` in `'LoadPointsFromMultiSweeps'` to `[0, 1, 2, 4]` by default,
where the first 3 dimensions refer to point coordinates, and the last refers to timestamp differences.

## Evaluation

An example to evaluate PointPillars with 8 GPUs with Lyft metrics is as follows.

```shell
bash ./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth 8 --eval bbox
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
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py work_dirs/pp-lyft/latest.pth 8 --out work_dirs/pp-lyft/results_challenge.pkl --format-only --eval-options 'jsonfile_prefix=work_dirs/pp-lyft/results_challenge' 'csv_savepath=results/pp-lyft/results_challenge.csv'
```

After generating the `work_dirs/pp-lyft/results_challenge.csv`, you can submit it to the Kaggle evaluation server. Please refer to the [offical website](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles) for more information.

We can also visualize the prediction results with our developed visualization tools. Please refer to the [visualization doc](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization) for more details.
