# KITTI 数据集

本页提供了有关在 MMDetection3D 中使用 KITTI 数据集的具体教程。

## 数据准备

您可以在[这里](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)下载 KITTI 3D 检测数据并解压缩所有 zip 文件。此外，您可以在[这里](https://download.openmmlab.com/mmdetection3d/data/train_planes.zip)下载道路平面信息，其在训练过程中作为一个可选项，用来提高模型的性能。道路平面信息由 [AVOD](https://github.com/kujason/avod) 生成，更多细节请参考[此处](https://github.com/kujason/avod/issues/19)。

像准备数据集的一般方法一样，建议将数据集根目录链接到 `$MMDETECTION3D/data`。

在我们处理之前，文件夹结构应按如下方式组织：

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
│   │   │   ├── planes (optional)
```

### 创建 KITTI 数据集

为了创建 KITTI 点云数据，首先需要加载原始的点云数据并生成相关的包含目标标签和标注框的数据标注文件，同时还需要为 KITTI 数据集生成每个单独的训练目标的点云数据，并将其存储在 `data/kitti/kitti_gt_database` 的 `.bin` 格式的文件中，此外，需要为训练数据或者验证数据生成 `.pkl` 格式的包含数据信息的文件。随后，通过运行下面的命令来创建最终的 KITTI 数据：

```bash
mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

# 下载数据划分
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt

python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti --with-plane
```

需要注意的是，如果您的本地磁盘没有充足的存储空间来存储转换后的数据，您可以通过改变 `--out-dir` 来指定其他任意的存储路径。如果您没有准备 `planes` 数据，您需要移除 `--with-plane` 标志。

处理后的文件夹结构应该如下：

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
│   ├── planes (optional)
├── kitti_gt_database
│   ├── xxxxx.bin
├── kitti_infos_train.pkl
├── kitti_infos_val.pkl
├── kitti_dbinfos_train.pkl
├── kitti_infos_test.pkl
├── kitti_infos_trainval.pkl
```

- `kitti_gt_database/xxxxx.bin`：训练数据集中包含在 3D 标注框中的点云数据。
- `kitti_infos_train.pkl`：训练数据集，该字典包含了两个键值：`metainfo` 和 `data_list`。`metainfo` 包含数据集的基本信息，例如 `categories`, `dataset` 和 `info_version`。`data_list` 是由字典组成的列表，每个字典（以下简称 `info`）包含了单个样本的所有详细信息。
  - info\['sample_idx'\]：该样本在整个数据集的索引。
  - info\['images'\]：多个相机捕获的图像信息。是一个字典，包含 5 个键值：`CAM0`, `CAM1`, `CAM2`, `CAM3`, `R0_rect`。
    - info\['images'\]\['R0_rect'\]：校准旋转矩阵，是一个 4x4 数组。
    - info\['images'\]\['CAM2'\]：包含 `CAM2` 相机传感器的信息。
      - info\['images'\]\['CAM2'\]\['img_path'\]：图像的文件名。
      - info\['images'\]\['CAM2'\]\['height'\]：图像的高。
      - info\['images'\]\['CAM2'\]\['width'\]：图像的宽。
      - info\['images'\]\['CAM2'\]\['cam2img'\]：相机到图像的变换矩阵，是一个 4x4 数组。
      - info\['images'\]\['CAM2'\]\['lidar2cam'\]：激光雷达到相机的变换矩阵，是一个 4x4 数组。
      - info\['images'\]\['CAM2'\]\['lidar2img'\]：激光雷达到图像的变换矩阵，是一个 4x4 数组。
    - info\['lidar_points'\]：是一个字典，包含了激光雷达点相关的信息。
      - info\['lidar_points'\]\['lidar_path'\]：激光雷达点云数据的文件名。
      - info\['lidar_points'\]\['num_pts_feats'\]：点的特征维度。
      - info\['lidar_points'\]\['Tr_velo_to_cam'\]：Velodyne 坐标到相机坐标的变换矩阵，是一个 4x4 数组。
      - info\['lidar_points'\]\['Tr_imu_to_velo'\]：IMU 坐标到 Velodyne 坐标的变换矩阵，是一个 4x4 数组。
    - info\['instances'\]：是一个字典组成的列表。每个字典包含单个实例的所有标注信息。对于其中的第 i 个实例，我们有：
      - info\['instances'\]\[i\]\['bbox'\]：长度为 4 的列表，以 (x1, y1, x2, y2) 的顺序表示实例的 2D 边界框。
      - info\['instances'\]\[i\]\['bbox_3d'\]：长度为 7 的列表，以 (x, y, z, l, h, w, yaw) 的顺序表示实例的 3D 边界框。
      - info\['instances'\]\[i\]\['bbox_label'\]：是一个整数，表示实例的 2D 标签，-1 代表忽略。
      - info\['instances'\]\[i\]\['bbox_label_3d'\]：是一个整数，表示实例的 3D 标签，-1 代表忽略。
      - info\['instances'\]\[i\]\['depth'\]：3D 边界框投影到相关图像平面的中心点的深度。
      - info\['instances'\]\[i\]\['num_lidar_pts'\]：3D 边界框内的激光雷达点数。
      - info\['instances'\]\[i\]\['center_2d'\]：3D 边界框投影的 2D 中心。
      - info\['instances'\]\[i\]\['difficulty'\]：KITTI 官方定义的困难度，包括简单、适中、困难。
      - info\['instances'\]\[i\]\['truncated'\]：从 0（非截断）到 1（截断）的浮点数，其中截断指的是离开检测图像边界的检测目标。
      - info\['instances'\]\[i\]\['occluded'\]：整数 (0,1,2,3) 表示目标的遮挡状态：0 = 完全可见，1 = 部分遮挡，2 = 大面积遮挡，3 = 未知。
      - info\['instances'\]\[i\]\['group_ids'\]：用于多部分的物体。
    - info\['plane'\]（可选）：地平面信息。

更多细节请参考 [kitti_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/kitti_converter.py) 和 [update_infos_to_v2.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/update_infos_to_v2.py)。

## 训练流程

下面展示了一个使用 KITTI 数据集进行 3D 目标检测的典型流程：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4, # x, y, z, intensity
        use_dim=4),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

- 数据增强：
  - `ObjectNoise`：对场景中的每个真实标注框目标添加噪音。
  - `RandomFlip3D`：对输入点云数据进行随机地水平翻转或者垂直翻转。
  - `GlobalRotScaleTrans`：对输入点云数据进行旋转。

## 评估

使用 8 个 GPU 以及 KITTI 指标评估的 PointPillars 的示例如下：

```shell
bash tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/latest.pth 8
```

## 度量指标

KITTI 官方使用全类平均精度（mAP）和平均方向相似度（AOS）来评估 3D 目标检测的性能，更多细节请参考[官方网站](http://www.cvlibs.net/datasets/kitti/eval_3dobject.php)和[论文](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf)。

MMDetection3D 采用相同的方法在 KITTI 数据集上进行评估，下面展示了一个评估结果的例子：

```
Car AP@0.70, 0.70, 0.70:
bbox AP:97.9252, 89.6183, 88.1564
bev  AP:90.4196, 87.9491, 85.1700
3d   AP:88.3891, 77.1624, 74.4654
aos  AP:97.70, 89.11, 87.38
Car AP@0.70, 0.50, 0.50:
bbox AP:97.9252, 89.6183, 88.1564
bev  AP:98.3509, 90.2042, 89.6102
3d   AP:98.2800, 90.1480, 89.4736
aos  AP:97.70, 89.11, 87.38
```

## 测试和提交

使用 8 个 GPU 在 KITTI 上测试 PointPillars 并生成对排行榜的提交的示例如下：

- 首先，你需要在你的配置文件中修改 `test_dataloader` 和 `test_evaluator` 字典，如下所示：

  ```python
  data_root = 'data/kitti/'
  test_dataloader = dict(
      dataset=dict(
          ann_file='kitti_infos_test.pkl',
          load_eval_anns=False,
          data_prefix=dict(pts='testing/velodyne_reduced')))
  test_evaluator = dict(
      ann_file=data_root + 'kitti_infos_test.pkl',
      format_only=True,
      pklfile_prefix='results/kitti-3class/kitti_results',
      submission_prefix='results/kitti-3class/kitti_results')
  ```

- 接下来，你可以运行如下测试脚本。

  ```shell
  ./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/latest.pth 8
  ```

在生成 `results/kitti-3class/kitti_results/xxxxx.txt` 后，您可以提交这些文件到 KITTI 官方网站进行基准测试，更多细节请参考 [KITTI 官方网站](http://www.cvlibs.net/datasets/kitti/index.php)。
