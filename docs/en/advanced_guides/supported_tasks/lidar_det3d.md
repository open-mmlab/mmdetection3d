# LiDAR-Based 3D Detection

LiDAR-based 3D detection is one of the most basic tasks supported in MMDetection3D.
It expects the given model to take any number of points with features collected by LiDAR as input, and predict the 3D bounding boxes and category labels for each object of interest.
Next, taking PointPillars on the KITTI dataset as an example, we will show how to prepare data, train and test a model on a standard 3D detection benchmark, and how to visualize and validate the results.

## Data Preparation

To begin with, we need to download the raw data and reorganize the data in a standard way presented in the [doc for data preparation](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/dataset_prepare.html).
Note that for KITTI, we need extra `.txt` files for data splits.

Due to different ways of organizing the raw data in different datasets, we typically need to collect the useful data information with a `.pkl` file.
So after getting all the raw data ready, we need to run the scripts provided in the `create_data.py` for different datasets to generate data infos.
For example, for KITTI we need to run:

```shell
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

Afterwards, the related folder structure should be as follows:

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
│   │   │   ├── velodyne_reduced
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
│   │   │   ├── velodyne_reduced
│   │   ├── kitti_gt_database
│   │   ├── kitti_infos_train.pkl
│   │   ├── kitti_infos_trainval.pkl
│   │   ├── kitti_infos_val.pkl
│   │   ├── kitti_infos_test.pkl
│   │   ├── kitti_dbinfos_train.pkl
```

## Training

Then let us train a model with provided configs for PointPillars.
You can basically follow the examples provided in this [tutorial](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/train_test.html) when training with different GPU settings.
Suppose we use 8 GPUs on a single machine with distributed training:

```shell
./tools/dist_train.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py 8
```

Note that `8xb6` in the config name refers to the training is completed with 8 GPUs and 6 samples on each GPU.
If your customized setting is different from this, sometimes you need to adjust the learning rate accordingly.
A basic rule can be referred to [here](https://arxiv.org/abs/1706.02677). We have supported `--auto-scale-lr` to
enable automatically scaling LR.

## Quantitative Evaluation

During training, the model checkpoints will be evaluated regularly according to the setting of `train_cfg = dict(val_interval=xxx)` in the config.
We support official evaluation protocols for different datasets.
For KITTI, the model will be evaluated with mean average precision (mAP) with Intersection over Union (IoU) thresholds 0.5/0.7 for 3 categories respectively.
The evaluation results will be printed in the command like:

```
Car AP@0.70, 0.70, 0.70:
bbox AP:98.1839, 89.7606, 88.7837
bev AP:89.6905, 87.4570, 85.4865
3d AP:87.4561, 76.7569, 74.1302
aos AP:97.70, 88.73, 87.34
Car AP@0.70, 0.50, 0.50:
bbox AP:98.1839, 89.7606, 88.7837
bev AP:98.4400, 90.1218, 89.6270
3d AP:98.3329, 90.0209, 89.4035
aos AP:97.70, 88.73, 87.34
```

In addition, you can also evaluate a specific model checkpoint after training is finished. Simply run scripts like the following:

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars/latest.pth 8
```

## Testing and Making a Submission

If you would like to only conduct inference or test the model performance on the online benchmark,
you need to specify the `submission_prefix` for corresponding evaluator,
e.g., add `test_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_test.pkl', format_only=True, pklfile_prefix='results/kitti-3class/kitti_results', submission_prefix='results/kitti-3class/kitti_results')` in the configuration then you can get the results file.
Please guarantee the `data_prefix` and `ann_file` in [info for testing](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/kitti-3d-3class.py#L117) in the config corresponds to the test set instead of validation set.
After generating the results, you can basically compress the folder and upload to the KITTI evaluation server.

## Qualitative Validation

MMDetection3D also provides versatile tools for visualization such that we can have an intuitive feeling of the detection results predicted by our trained models.
You can either set the `--show` option to visualize the detection results online during evaluation,
or using `tools/misc/visualize_results.py` for offline visualization.
Besides, we also provide scripts `tools/misc/browse_dataset.py` to visualize the dataset without inference.
Please refer more details in the [doc for visualization](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/visualization.html).
