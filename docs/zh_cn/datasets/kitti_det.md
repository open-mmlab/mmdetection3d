# 3D 目标检测 KITTI 数据集

本页提供了有关在 MMDetection3D 中使用 KITTI 数据集的具体教程。

**注意**：此教程目前仅适用于基于雷达和多模态的 3D 目标检测的相关方法，与基于单目图像的 3D 目标检测相关的内容会在之后进行补充。

## 数据准备

您可以在[这里](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)下载 KITTI 3D 检测数据并解压缩所有 zip 文件。此外，您可以在[这里](https://download.openmmlab.com/mmdetection3d/data/train_planes.zip)下载道路平面信息，其在训练过程中作为一个可选项，用来提高模型的性能。道路平面信息由 [AVOD](https://github.com/kujason/avod) 生成，你可以在[这里](https://github.com/kujason/avod/issues/19)查看更多细节。

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

# Download data split
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt

python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

需要注意的是，如果您的本地磁盘没有充足的存储空间来存储转换后的数据，您可以通过改变 `out-dir` 来指定其他任意的存储路径。

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

其中的各项文件的含义如下所示：

- `kitti_gt_database/xxxxx.bin`： 训练数据集中包含在 3D 标注框中的点云数据
- `kitti_infos_train.pkl`：训练数据集的信息，其中每一帧的信息包含下面的内容：
    - info['point_cloud']: {'num_features': 4, 'velodyne_path': velodyne_path}.
    - info['annos']: {
        - 位置：其中 x,y,z 为相机参考坐标系下的目标的底部中心（单位为米），是一个尺寸为 Nx3 的数组
        - 维度: 目标的高、宽、长（单位为米），是一个尺寸为 Nx3 的数组
        - 旋转角：相机坐标系下目标绕着 Y 轴的旋转角 ry，其取值范围为 [-pi..pi] ，是一个尺寸为 N 的数组
        - 名称：标准框所包含的目标的名称，是一个尺寸为 N 的数组
        - 困难度：kitti 官方所定义的困难度，包括 简单，适中，困难
        - 组别标识符：用于多部件的目标
        }
    - (optional) info['calib']: {
        - P0：校对后的 camera0 投影矩阵，是一个 3x4 数组
        - P1：校对后的 camera1 投影矩阵，是一个 3x4 数组
        - P2：校对后的 camera2 投影矩阵，是一个 3x4 数组
        - P3：校对后的 camera3 投影矩阵，是一个 3x4 数组
        - R0_rect：校准旋转矩阵，是一个 4x4 数组
        - Tr_velo_to_cam：从 Velodyne 坐标到相机坐标的变换矩阵，是一个 4x4 数组
        - Tr_imu_to_velo：从 IMU 坐标到 Velodyne 坐标的变换矩阵，是一个 4x4 数组
    }
    - (optional) info['image']:{'image_idx': idx, 'image_path': image_path, 'image_shape', image_shape}.

**注意**：其中的 info['annos'] 中的数据均位于相机参考坐标系中，更多的细节请参考[此处](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)。

获取 kitti_infos_xxx.pkl 和 kitti_infos_xxx_mono3d.coco.json 的核心函数分别为 [get_kitti_image_info](https://github.com/open-mmlab/mmdetection3d/blob/7873c8f62b99314f35079f369d1dab8d63f8a3ce/tools/data_converter/kitti_data_utils.py#L140) 和 [get_2d_boxes](https://github.com/open-mmlab/mmdetection3d/blob/7873c8f62b99314f35079f369d1dab8d63f8a3ce/tools/data_converter/kitti_converter.py#L378).

## 训练流程

下面展示了一个使用 KITTI 数据集进行 3D 目标检测的典型流程：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4, # x, y, z, intensity
        use_dim=4, # x, y, z, intensity
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

- 数据增强：
    - `ObjectNoise`：对场景中的每个真实标注框目标添加噪音。
    - `RandomFlip3D`：对输入点云数据进行随机地水平翻转或者垂直翻转。
    - `GlobalRotScaleTrans`：对输入点云数据进行旋转。

## 评估

使用 8 个 GPU 以及 KITTI 指标评估的 PointPillars 的示例如下：

```shell
bash tools/dist_test.sh configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth 8 --eval bbox
```

## 度量指标

KITTI 官方使用全类平均精度（mAP）和平均方向相似度（AOS）来评估 3D 目标检测的性能，请参考[官方网站](http://www.cvlibs.net/datasets/kitti/eval_3dobject.php)和[论文](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf)获取更多细节。

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

```shell
mkdir -p results/kitti-3class

./tools/dist_test.sh configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth 8 --out results/kitti-3class/results_eval.pkl --format-only --eval-options 'pklfile_prefix=results/kitti-3class/kitti_results' 'submission_prefix=results/kitti-3class/kitti_results'
```

在生成 `results/kitti-3class/kitti_results/xxxxx.txt` 后，您可以提交这些文件到 KITTI 官方网站进行基准测试，请参考 [KITTI 官方网站]((http://www.cvlibs.net/datasets/kitti/index.php))获取更多细节。
