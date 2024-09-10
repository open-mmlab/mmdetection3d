# SemanticKITTI Dataset

This page provides specific tutorials about the usage of MMDetection3D for SemanticKITTI dataset.

## Prepare dataset

You can download SemanticKITTI dataset [HERE](http://semantic-kitti.org/dataset.html#download) and unzip all zip files.

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.

The folder structure should be organized as follows before our processing.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 22
```

SemanticKITTI dataset contains 23 sequences, where \[0-7\], \[9-10\] are used as training set (about 19k training samples), sequence 8 as validation set (about 4k validation samples) and \[11-22\] as test set (about 20k test samples). Each sequence contains velodyne and labels folders for LIDAR point cloud data and segmentation annotations (where the high 16 bits store the instance segmentation annotations and the low 16 bits store the semantic segmentation annotations), respectively.

### Create SemanticKITTI Dataset

We support scripts that generate dataset information for training and testing. Create `.pkl` info by running:

```bash
python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti
```

The folder structure after processing should be as below

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 22
│   │   ├── semantickitti_infos_test.pkl
│   │   ├── semantickitti_infos_train.pkl
│   │   ├── semantickitti_infos_val.pkl
```

- `semantickitti_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:
  - info\['sample_id'\]: The index of this sample in the whole dataset.
  - info\['lidar_points'\]: A dict containing all the information related to the lidar points.
    - info\['lidar_points'\]\['lidar_path'\]: The filename of the lidar point cloud data.
    - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of point.
  - info\['pts_semantic_mask_pth'\]: The path of 3D semantic segmentation annotation file.

Please refer to [semantickitti_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/semantickitti_converter.py) and [update_infos_to_v2.py ](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/update_infos_to_v2.py) for more details.

## Train pipeline

A typical train pipeline of 3D segmentation on SemanticKITTI is as below:

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
```

- Data augmentation:
  - `RandomFlip3D`: randomly flip input point cloud horizontally or vertically.
  - `GlobalRotScaleTrans`: rotate/scale/transform input point cloud.

## Evaluation

An example to evaluate MinkUNet with 8 GPUs with semantickitti metrics is as follows:

```shell
bash tools/dist_test.sh configs/minkunet/minkunet_w32_8xb2-15e_semantickitti.py checkpoints/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth 8
```

## Metrics

Typically mean intersection over union (mIoU) is used for evaluation on Semantickitti. In detail, we first compute IoU for multiple classes and then average them to get mIoU, please refer to [seg_eval.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/evaluation/functional/seg_eval.py).

An example of printed evaluation results is as follows:

| classes | car    | bicycle | motorcycle | truck  | bus    | person | bicyclist | motorcyclist | road   | parking | sidewalk | other-ground | building | fence  | vegetation | trunck | terrian | pole   | traffic-sign | miou   | acc    | acc_cls |
| ------- | ------ | ------- | ---------- | ------ | ------ | ------ | --------- | ------------ | ------ | ------- | -------- | ------------ | -------- | ------ | ---------- | ------ | ------- | ------ | ------------ | ------ | ------ | ------- |
| results | 0.9687 | 0.1908  | 0.6313     | 0.8580 | 0.6359 | 0.6818 | 0.8444    | 0.0002       | 0.9353 | 0.4854  | 0.8106   | 0.0024       | 0.9050   | 0.6111 | 0.8822     | 0.6605 | 0.7493  | 0.6442 | 0.4837       | 0.6306 | 0.9202 | 0.6924  |
