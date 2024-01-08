# 数据预处理

## 在数据预处理前

我们推荐用户将数据集的路径软链接到 `$MMDETECTION3D/data`。如果你的文件夹结构和以下所展示的结构不一致，你可能需要改变配置文件中相应的数据路径。

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
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (训练数据)
│   │   │   ├── lidar (训练激光雷达)
│   │   │   ├── images (训练图片)
│   │   │   ├── maps (训练地图)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (测试数据)
│   │   │   ├── lidar (测试激光雷达)
│   │   │   ├── images (测试图片)
│   │   │   ├── maps (测试地图)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
│   ├── s3dis
│   │   ├── meta_data
│   │   ├── Stanford3dDataset_v1.2_Aligned_Version
│   │   ├── collect_indoor3d_data.py
│   │   ├── indoor3d_util.py
│   │   ├── README.md
│   ├── scannet
│   │   ├── meta_data
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
│   ├── sunrgbd
│   │   ├── OFFICIAL_SUNRGBD
│   │   ├── matlab
│   │   ├── sunrgbd_data.py
│   │   ├── sunrgbd_utils.py
│   │   ├── README.md

```

## 数据下载和预处理

### KITTI

在[这里](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)下载 KITTI 的 3D 检测数据。通过运行以下指令对 KITTI 数据进行预处理：

```bash
mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

# 下载数据划分文件
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
```

然后通过运行以下指令生成信息文件：

```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

在使用 slurm 的环境下，用户需要使用下面的指令：

```bash
sh tools/create_data.sh <partition> kitti
```

**小贴士**：

- **现成的标注文件**：我们已经提供了离线处理好的 [KITTI 标注文件](#数据集标注文件列表)。您直接下载他们并放到 `data/kitti/` 目录下。然而，如果你想在点云检测方法中使用 `ObjectSample` 这一数据增强，你可以再额外使用以下命令来生成物体标注框数据库：

```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti --only-gt-database
```

### Waymo

在[这里](https://waymo.com/open/download/)下载 Waymo 公开数据集 1.4.1 版本，在[这里](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing)下载其数据划分文件。然后，将 `.tfrecord` 文件置于 `data/waymo/waymo_format/` 目录下的相应位置，并将数据划分的 `.txt` 文件置于 `data/waymo/kitti_format/ImageSets` 目录下。在[这里](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects)下载验证集的真实标签（`.bin` 文件）并将其置于 `data/waymo/waymo_format/`。提示：你可以使用 `gsutil` 来用命令下载大规模的数据集。更多细节请参考此[工具](https://github.com/RalphMao/Waymo-Dataset-Tool)。完成以上各步后，可以通过运行以下指令对 Waymo 数据进行预处理：

```bash
# TF_CPP_MIN_LOG_LEVEL=3 will disable all logging output from TensorFlow.
# The number of `--workers` depends on the maximum number of cores in your CPU.
TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

注意:

- 如果你的硬盘空间大小不足以存储转换后的数据，你可以将 `--out-dir` 参数设定为别的路径。你只需要记得在那个路径下创建文件夹并下载数据，然后在数据预处理完成后将其链接回 `data/waymo/kitti_format` 即可。

**小贴士**：

- **现成的标注文件**: 我们已经提供了离线处理好的 [Waymo 标注文件](#数据集标注文件列表)。您直接下载他们并放到 `data/waymo/kitti_format/` 目录下。然而，您还是需要自己使用上面的脚本将 Waymo 的原始数据还需要转成 kitti 格式。

- **Waymo-mini**： 如果你只是为了验证某些方法或者 debug, 你可以使用我们提供的 [Waymo-mini](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz)。它只包含原始数据集中训练集中的 2 个 segments 和 验证集中的 1 个 segment。您只需要下载并且解压到 `data/waymo_mini/`，即可使用它：

  ```bash
  tar -xzvf waymo_mini.tar.gz -C ./data/waymo_mini
  ```

### NuScenes

在[这里](https://www.nuscenes.org/download)下载 nuScenes 数据集 1.0 版本的完整数据文件。通过运行以下指令对 nuScenes 数据进行预处理：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

**小贴士**：

- **现成的标注文件**：我们已经提供了离线处理好的 [NuScenes 标注文件](#数据集标注文件列表)。您直接下载他们并放到 `data/nuscenes/` 目录下。然而，如果你想在点云检测方法中使用 `ObjectSample` 这一数据增强，你可以再额外使用以下命令来生成物体标注框数据库：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --only-gt-database
```

### Lyft

在[这里](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data)下载 Lyft 3D 检测数据。通过运行以下指令对 Lyft 数据进行预处理：

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/data_converter/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

注意，为了文件结构的清晰性，我们遵从了 Lyft 数据原先的文件夹名称。请按照上面展示出的文件结构对原始文件夹进行重命名。同样值得注意的是，第二行命令的目的是为了修复一个损坏的激光雷达数据文件。更多细节请参考[该讨论](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000)。

### SemanticKITTI

在[这里](http://semantic-kitti.org/dataset.html#download)下载 SemanticKITTI 数据集并解压所有文件。通过运行以下指令对 SemanticKITTI 数据进行预处理：

```bash
python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti
```

**小贴士**：

- **现成的标注文件**. 我们已经提供了离线处理好的 [SemanticKITTI 标注文件](#数据集标注文件列表)。您直接下载他们并放到 `data/semantickitti` 目录下。

### S3DIS、ScanNet 和 SUN RGB-D

请参考 S3DIS [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/s3dis/README.md) 文件以对其进行数据预处理。

请参考 ScanNet [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/scannet/README.md) 文件以对其进行数据预处理。

请参考 SUN RGB-D [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/sunrgbd/README.md) 文件以对其进行数据预处理。

**小贴士**：对于 S3DIS, ScanNet 和 SUN RGB-D 数据集，我们已经提供了离线处理好的 [标注文件](#数据集标注文件列表)。您可以直接下载他们并放到 `data/${DATASET}/` 目录下。然而，您还是需要自己利用我们的脚本来生成点云文件以及语义掩膜文件(如果该数据集有的话)。

### 自定义数据集

关于如何使用自定义数据集，请参考[自定义数据集](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/zh_cn/advanced_guides/customize_dataset.md)。

### 更新数据信息

如果你之前已经使用 v1.0.0rc1-v1.0.0rc4 版的 mmdetection3d 创建数据信息，现在你想使用最新的 v1.1.0 版 mmdetection3d，你需要更新数据信息文件。

```bash
python tools/dataset_converters/update_infos_to_v2.py --dataset ${DATA_SET} --pkl-path ${PKL_PATH} --out-dir ${OUT_DIR}
```

- `--dataset`：数据集名。
- `--pkl-path`：指定数据信息 pkl 文件路径。
- `--out-dir`：输出数据信息 pkl 文件目录。

例如：

```bash
python tools/dataset_converters/update_infos_to_v2.py --dataset kitti --pkl-path ./data/kitti/kitti_infos_trainval.pkl --out-dir ./data/kitti
```

### 数据集标注文件列表

我们提供了离线生成好的数据集标注文件以供参考。为了方便，您也可以直接使用他们。

|                                                  数据集                                                   |                                                                                                              训练集标注文件                                                                                                               |                                                                                                           验证集标注文件                                                                                                           |                                                                                                                 测试集标注文件                                                                                                                  |
| :-------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                   KITTI                                                   |                                                                  [kitti_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/kitti/kitti_infos_train.pkl)                                                                   |                                                                 [kitti_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/kitti/kitti_infos_val.pkl)                                                                 |                                                                        [kitti_infos_test](https://download.openmmlab.com/mmdetection3d/data/kitti/kitti_infos_test.pkl)                                                                         |
|                                                 NuScenes                                                  | [nuscenes_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl) [nuscenes_mini_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_mini_infos_train.pkl) | [nuscenes_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl)  [nuscenes_mini_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_mini_infos_val.pkl) |                                                                                                                                                                                                                                                 |
|                                                   Waymo                                                   |                                                         [waymo_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_infos_train.pkl)                                                          |                                                        [waymo_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_infos_val.pkl)                                                        | [waymo_infos_test.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo/waymo_infos_test.pkl)   [waymo_infos_test_cam_only.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_infos_test_cam_only.pkl) |
| [Waymo-mini](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz) |                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                 |
|                                                 SUN RGB-D                                                 |                                                               [sunrgbd_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/sunrgbd/sunrgbd_infos_train.pkl)                                                                |                                                              [sunrgbd_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/sunrgbd/sunrgbd_infos_val.pkl)                                                              |                                                                                                                                                                                                                                                 |
|                                                  ScanNet                                                  |                                                               [scannet_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/scannet/scannet_infos_train.pkl)                                                                |                                                              [scannet_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/scannet/scannet_infos_val.pkl)                                                              |                                                                   [scannet_infos_test.pkl](https://download.openmmlab.com/mmdetection3d/data/scannet/scannet_infos_test.pkl)                                                                    |
|                                               SemanticKitti                                               |                                                      [semantickitti_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/semantickitti/semantickitti_infos_train.pkl)                                                       |                                                     [semantickitti_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/semantickitti/semantickitti_infos_val.pkl)                                                     |                                                          [semantickitti_infos_test.pkl](https://download.openmmlab.com/mmdetection3d/data/semantickitti/semantickitti_infos_test.pkl)                                                           |
