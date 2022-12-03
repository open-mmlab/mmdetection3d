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

### Waymo

在[这里](https://waymo.com/open/download/)下载 Waymo 公开数据集 1.2 版本，在[这里](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing)下载其数据划分文件。然后，将 `.tfrecord` 文件置于 `data/waymo/waymo_format/` 目录下的相应位置，并将数据划分的 `.txt` 文件置于 `data/waymo/kitti_format/ImageSets` 目录下。在[这里](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects)下载验证集的真实标签（`.bin` 文件）并将其置于 `data/waymo/waymo_format/`。提示：你可以使用 `gsutil` 来用命令下载大规模的数据集。更多细节请参考此[工具](https://github.com/RalphMao/Waymo-Dataset-Tool)。完成以上各步后，可以通过运行以下指令对 Waymo 数据进行预处理：

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```

注意:

- 如果你的硬盘空间大小不足以存储转换后的数据，你可以将 `--out-dir` 参数设定为别的路径。你只需要记得在那个路径下创建文件夹并下载数据，然后在数据预处理完成后将其链接回 `data/waymo/kitti_format` 即可。

- 如果你想在 Waymo 上进行更快的评估，你可以下载已经预处理好的[元信息文件](https://download.openmmlab.com/mmdetection3d/data/waymo/idx2metainfo.pkl)并将其放置在 `data/waymo/waymo_format/` 目录下。接着，你可以按照以下来修改数据集的配置：

  ```python
  val_evaluator = dict(
      type='WaymoMetric',
      ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
      waymo_bin_file='./data/waymo/waymo_format/gt.bin',
      data_root='./data/waymo/waymo_format',
      file_client_args=file_client_args,
      convert_kitti_format=True,
      idx2metainfo='data/waymo/waymo_format/idx2metainfo.pkl'
      )
  ```

  目前这种方式仅限于纯点云检测任务。

### NuScenes

在[这里](https://www.nuscenes.org/download)下载 nuScenes 数据集 1.0 版本的完整数据文件。通过运行以下指令对 nuScenes 数据进行预处理：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### Lyft

在[这里](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data)下载 Lyft 3D 检测数据。通过运行以下指令对 Lyft 数据进行预处理：

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/data_converter/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

注意，为了文件结构的清晰性，我们遵从了 Lyft 数据原先的文件夹名称。请按照上面展示出的文件结构对原始文件夹进行重命名。同样值得注意的是，第二行命令的目的是为了修复一个损坏的激光雷达数据文件。更多细节请参考[该讨论](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000)。

### S3DIS、ScanNet 和 SUN RGB-D

请参考 S3DIS [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/s3dis/README.md) 文件以对其进行数据预处理。

请参考 ScanNet [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/scannet/README.md) 文件以对其进行数据预处理。

请参考 SUN RGB-D [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/sunrgbd/README.md) 文件以对其进行数据预处理。

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
