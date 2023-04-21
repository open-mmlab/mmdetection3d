# Train with Customized Datasets

In this note, you will know how to train and test predefined models with customized datasets. We use the Waymo dataset as an example to describe the whole process.

The basic steps are as below:

1. Prepare the customized dataset
2. Prepare a config
3. Train, test, inference models on the customized dataset.

## Prepare the customized dataset

There are three ways to support a new dataset in MMDetection3D:

1. reorganize the dataset into existing format.
2. reorganize the dataset into a standard format.
3. implement a new dataset.

Usually we recommend to use the first two methods which are usually easier than the third.

In this note, we give an example for converting the data into KITTI format, you can refer to this to reorganize your dataset into kitti format. About the standard format dataset, and you can refer to [customize_dataset.md](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/en/advanced_guides/customize_dataset.md).

**Note**: We take Waymo as the example here considering its format is totally different from other existing formats. For other datasets using similar methods to organize data, like Lyft compared to nuScenes, it would be easier to directly implement the new data converter (for the second approach above) instead of converting it to another format (for the first approach above).

### KITTI dataset format

Firstly, the raw data for 3D object detection from KITTI are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation/testing set, `calib` contains calibration information files, `image_2` and `velodyne` include image data and point cloud data, and `label_2` includes label files for 3D detection.

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

Specific annotation format is described in the official object development [kit](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip). For example, it consists of the following labels:

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```

Assume we use the Waymo dataset.

After downloading the data, we need to implement a function to convert both the input data and annotation format into the KITTI style. Then we can implement `WaymoDataset` inherited from `KittiDataset` to load the data and perform training, and implement `WaymoMetric` inherited from `KittiMetric` for evaluation.

Specifically, we implement a waymo [converter](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/waymo_converter.py) to convert Waymo data into KITTI format and a waymo dataset [class](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/datasets/waymo_dataset.py) to process it, in addition need to add a waymo [metric](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/evaluation/metrics/waymo_metric.py) to evaluate results. Because we preprocess the raw data and reorganize it like KITTI, the dataset class could be implemented more easily by inheriting from KittiDataset. Regarding the dataset evaluation metric, because Waymo has its own evaluation approach, we need further implement a new Waymo metric; more about the metric could refer to [metric_and_evaluator.md](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/metric_and_evaluator.md). Afterward, users can successfully convert the data format and use `WaymoDataset` to train and evaluate the model by `WaymoMetric`.

For more details about the intermediate results of preprocessing of Waymo dataset, please refer to its [waymo_det.md](https://mmdetection3d.readthedocs.io/en/latest/datasets/waymo_det.html).

## Prepare a config

The second step is to prepare configs such that the dataset could be successfully loaded. In addition, adjusting hyperparameters is usually necessary to obtain decent performance in 3D detection.

Suppose we would like to train PointPillars on Waymo to achieve 3D detection for 3 classes, vehicle, cyclist and pedestrian, we need to prepare dataset config like [this](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/waymoD5-3d-3class.py), model config like [this](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/models/pointpillars_hv_secfpn_waymo.py) and combine them like [this](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py), compared to KITTI [dataset config](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/kitti-3d-3class.py), [model config](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/models/pointpillars_hv_secfpn_kitti.py) and [overall](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py).

## Train a new model

To train a model with the new config, you can simply run

```shell
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py
```

For more detailed usages, please refer to the [Case 1](https://mmdetection3d.readthedocs.io/en/latest/1_exist_data_model.html).

## Test and inference

To test the trained model, you can simply run

```shell
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class/latest.pth
```

**Note**: To use Waymo evaluation protocol, you need to follow the [tutorial](https://mmdetection3d.readthedocs.io/en/latest/datasets/waymo_det.html) and prepare files related to metrics computation as official instructions.

For more detailed usages for test and inference, please refer to the [Case 1](https://mmdetection3d.readthedocs.io/en/latest/1_exist_data_model.html).
