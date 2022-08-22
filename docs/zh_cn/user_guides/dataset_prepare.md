# 数据预处理

## 在数据预处理前

我们推荐用户将数据集的路径软链接到 `$MMDETECTION3D/data`。
如果你的文件夹结构和以下所展示的结构相异，你可能需要改变配置文件中相应的数据路径。

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

python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

### Waymo

在[这里](https://waymo.com/open/download/)下载 Waymo 公开数据集1.2版本，在[这里](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing)下载其数据划分文件。
然后，将 tfrecord 文件置于 `data/waymo/waymo_format/` 目录下的相应位置，并将数据划分的 txt 文件置于 `data/waymo/kitti_format/ImageSets` 目录下。
在[这里](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects)下载验证集的真实标签 (bin 文件) 并将其置于 `data/waymo/waymo_format/`。
提示，你可以使用 `gsutil` 来用命令下载大规模的数据集。你可以参考这个[工具](https://github.com/RalphMao/Waymo-Dataset-Tool)来获取更多实现细节。
完成以上各步后，可以通过运行以下指令对 Waymo 数据进行预处理：

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```

注意，如果你的硬盘空间大小不足以存储转换后的数据，你可以将 `out-dir` 参数设定为别的路径。
你只需要记得在那个路径下创建文件夹并下载数据，然后在数据预处理完成后将其链接回 `data/waymo/kitti_format` 即可。

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

注意，为了文件结构的清晰性，我们遵从了 Lyft 数据原先的文件夹名称。请按照上面展示出的文件结构对原始文件夹进行重命名。
同样值得注意的是，第二行命令的目的是为了修复一个损坏的激光雷达数据文件。请参考[这一](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000)讨论来获取更多细节。

### S3DIS、ScanNet 和 SUN RGB-D

请参考 S3DIS [README](https://github.com/open-mmlab/mmdetection3d/blob/master/data/s3dis/README.md/) 文件以对其进行数据预处理。

请参考 ScanNet [README](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/) 文件以对其进行数据预处理。

请参考 SUN RGB-D [README](https://github.com/open-mmlab/mmdetection3d/blob/master/data/sunrgbd/README.md/) 文件以对其进行数据预处理。

### 自定义数据集

关于如何使用自定义数据集，请参考[教程 2: 自定义数据集](https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html)。
