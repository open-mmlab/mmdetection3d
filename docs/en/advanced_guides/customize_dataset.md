# Customize Datasets

In this note, you will know how to train and test predefined models with customized datasets. We use the Waymo dataset as an example to describe the whole process.

The basic steps are as below:

1. Prepare data
2. Prepare a config
3. Train, test, inference models on the customized dataset.

## Data Preparation

The ideal situation is that we can reorganize the customized raw data and convert the annotation format into KITTI style. However, considering some calibration files and 3D annotations in KITTI format are difficult to obtain for customized datasets, we introduce the basic data format in the doc.

### Basic Data Format

#### Point cloud Format

Currently, we only support `.bin` format point cloud training and inference, before training on your own datasets, you need to transform your point cloud format to `.bin` file. The common point cloud data formats include `.pcd` and `.las`, we list some open-source tools for reference.

1. Convert pcd to bin: https://github.com/leofansq/Tools_RosBag2KITTI
2. Convert las to bin: The common conversion path is las -> pcd -> bin, and the conversion from las -> pcd can be achieved through [this tool](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor).

#### Label Format

The most basic information: 3D bounding box and category label of each scene need to be contained in annotation `.txt` file. Each line represents a 3D box in a given scene as follow:

```
# format: [x, y, z, dx, dy, dz, yaw, category_name]
1.23 1.42 0.23 3.96 1.65 1.55 1.56 Car
3.51 2.15 0.42 1.05 0.87 1.86 1.23 Pedestrian
...
```

The 3D Box should be stored in unified 3D coordinates.

#### Calibration Format

During data collection, we will have multiple lidars and cameras with different sensor setup. For the point cloud data collected by each lidar, they are usually fused and  converted to a certain LiDAR coordinate, So typically the calibration information file should contain the intrinsic matrix of each camera and the transformation extrinsic matrix from the lidar to each camera in calibration `.txt` file, while `Px` represents the intrinsic matrix of `camera_x` and `lidar2camx` represents the transformation extrinsic matrix from the `lidar` to `camera_x`.

```
P0
P1
P2
P3
P4
...
lidar2cam0
lidar2cam1
lidar2cam2
lidar2cam3
lidar2cam4
...
```

### Raw Data Structure

#### LiDAR-Based 3D Detection

The raw data for LiDAR-Based 3D object detection are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation set, `points` include point cloud data which are supposed to be stored in `.bin` format and `labels` includes label files for 3D detection.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### Vision-Based 3D Detection

The raw data for Vision-Based 3D object detection are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation set, `images` contains the images from different cameras, for example, images from `camera_x` need to be placed in `images\images_x`. `calibs` contains calibration information files which store the camera intrinsic matrix of each camera, and `labels` includes label files for 3D detection.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### Multi-Modality 3D Detection

The raw data for Multi-Modality 3D object detection are typically organized as follows. Different from Vision-based 3D Object detection, calibration information files in `calibs` store the camera intrinsic matrix of each camera and extrinsic matrix.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### LiDAR-Based 3D Semantic Segmentation

The raw data for LiDAR-Based 3D semantic segmentation are typically organized as follows, where `ImageSets` contains split files indicating which files belong to training/validation set, `points` includes point cloud data, and `semantic_mask` includes point-level label.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── semantic_mask
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
```

### Data Converter

Once you prepared the raw data following our instruction, you can directly use the following command to generate training/validation information files.

```
python tools/create_data.py base --root-path ./data/custom --out-dir ./data/custom
```

## An example of customized dataset

Once we finish data preparation, we can create a new dataset in `mmdet3d/datasets/my_dataset.py` to load the data.

```
import mmengine

from mmdet.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class MyDataset(Det3DDataset):

    METAINFO = {
       'CLASSES': ('person', 'bicycle', 'car', 'motorcycle')
    }

    def parse_ann_info(self, info):
        """Get annotation info according to the given index.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        lidar2cam = np.array(info['images']['CAM0']['lidar2cam'])
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d)
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
```

## Prepare a config

The second step is to prepare configs such that the dataset could be successfully loaded. In addition, adjusting hyperparameters is usually necessary to obtain decent performance in 3D detection.

Suppose we would like to train PointPillars on Waymo to achieve 3D detection for 3 classes, vehicle, cyclist and pedestrian, we need to prepare dataset config like [this](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/waymoD5-3d-3class.py), model config like [this](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/hv_pointpillars_secfpn_waymo.py) and combine them like [this](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py), compared to KITTI [dataset config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/kitti-3d-3class.py), [model config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/hv_pointpillars_secfpn_kitti.py) and [overall](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py).

## Train a new model

To train a model with the new config, you can simply run

```shell
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py
```

For more detailed usages, please refer to the [Case 1](https://mmdetection3d.readthedocs.io/en/latest/1_exist_data_model.html).

## Test and inference

To test the trained model, you can simply run

```shell
python tools/test.py configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py work_dirs/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/latest.pth --eval waymo
```

**Note**: To use Waymo evaluation protocol, you need to follow the [tutorial](https://mmdetection3d.readthedocs.io/en/latest/datasets/waymo_det.html) and prepare files related to metrics computation as official instructions.

For more detailed usages for test and inference, please refer to the [Case 1](https://mmdetection3d.readthedocs.io/en/latest/1_exist_data_model.html).
