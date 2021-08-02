# KITTI Dataset

This page provides specific tutorials about the usage of MMDetection3D for kitti dataset.

## Prepare dataset

You can download KITTI 3D detection data [HERE](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and unzip all zip files.

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.

The folder structure should be organized as follows before our processing.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
```

### Create KITTI dataset

By Creating KITTI point cloud data, we load the raw point cloud data and generate the relevant annotations including object labels and bounding boxes. We also generate all single training objects' point cloud in KITTI dataset and save them as `.bin` files in `data/kitti/kitti_gt_database`. Meanwhile `.pkl` info files are also generated for train or validation. Subsequently, create KITTI data by running

```bash
mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

# Download data split
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt

python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

Note that if your local disk does not have enough space for saving converted data, you can change the `out-dir` to anywhere else.

The folder structure after process should be as below
```
kitti
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   ├── val.txt
├── testing
│   ├── calib
│   ├── image_2
│   ├── velodyne
│   ├── velodyne_reduced
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
│   ├── velodyne
│   ├── velodyne_reduced
├── kitti_gt_database
│   ├── xxxxx.bin
├── kitti_infos_train.pkl
├── kitti_infos_val.pkl
├── kitti_dbinfos_train.pkl
├── kitti_infos_test.pkl
├── kitti_infos_trainval.pkl
├── kitti_infos_train_mono3d.coco.json
├── kitti_infos_trainval_mono3d.coco.json
├── kitti_infos_test_mono3d.coco.json
├── kitti_infos_val_mono3d.coco.json
```

- `kitti_gt_database/xxxxx.bin`: point cloud data included in each 3D bounding box of the training dataset
- `kitti_infos_train.pkl`: training dataset infos, each frame info contains following details:
    - info['point_cloud']: {'num_features': 4, 'velodyne_path': velodyne_path}.
    - info['annos']: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    - (optional) info['calib']: the calibration info.
    - (optional) info['image']:{'image_idx': idx, 'image_path': image_path, 'image_shape', image_shape}.
**Note:** the info['annos'] is in the referenced camera coordinate system. More details please refer to [this](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)

## Train pipeline

A typical train pipeline of KITTI for 3d detection is as below.

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

- Data augmentation:
    - `ObjectNoise`: apply noise to each GT objects in the scene.
    - `RandomFlip3D`: randomly flip input point cloud horizontally or vertically.
    - `GlobalRotScaleTrans`: rotate input point cloud.

## Evaluation
An example to evaluate PointPillars with 8 GPUs with kitti metrics is as follows

```shell
bash tools/dist_test.sh configs/pointrcnn/pointrcnn_2x8_kitti-3d-3classes.py work_dirs/pointrcnn_2x8_kitti-3d-3classes/latest.pth 8 --eval bbox
```

## Testing and make a submission

An example to test PointPillars on kitti with 8 GPUs and generate a submission to the leaderboard is as follows

```shell
mkdir -p results/kitti-3class

./tools/dist_test.sh configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth 8 --out results/kitti-3class/results_eval.pkl --format-only --eval-options 'pklfile_prefix=results/kitti-3class/kitti_results' 'submission_prefix=results/kitti-3class/kitti_results'
```

After generating `results/kitti-3class/kitti_results/xxxxx.txt` files, you can submit these files to kitti benchmark, More informations please refer to the [KITTI offical website](http://www.cvlibs.net/datasets/kitti/index.php)
