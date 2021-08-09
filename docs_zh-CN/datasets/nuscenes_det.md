# 3D目标检测NuScenes数据集

本页提供了有关使用在 MMDetection3D 中使用 nuScenes 数据集的具体教程。

## 准备之前

您可以在[这里](https://www.nuscenes.org/download)下载 nuScenes 3D 检测数据并解压缩所有 zip 文件。

像准备数据集的一般方法一样，建议将数据集根目录链接 `$MMDETECTION3D/data`。

在我们处理之前，文件夹结构应按如下方式组织。

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
```

## 数据准备

我们通常需要使用特定样式的 .pkl 或 .json 文件来组织有用的数据信息，例如用于组织图像及其注释的 coco 样式。
要为 nuScenes 准备这些文件，请运行以下命令：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

处理后的文件夹结构应该如下

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
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_trainval.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_trainval_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json
```

这里，.pkl 文件一般用于涉及点云的方法，coco 风格的 .json 文件更适合基于图像的方法，例如基于图像的 2D 和 3D 检测。
接下来，我们将详细说明这些信息文件中记录的详细信息。
