# 仅lidar 数据集

本页提供了有关在 MMDetection3D 中使用 仅lidar 数据集的具体教程。

## 数据准备

像准备数据集的一般方法一样，建议将数据集根目录链接到 `$MMDETECTION3D/data`。

在我们处理之前，文件夹结构应按如下方式组织：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── label_2
│   │   │   ├── velodyne
```

### 创建 custom 数据集

为了创建 纯lidar 点云数据，首先需要加载原始的点云数据并生成相关的包含目标标签和标注框的数据标注文件，同时还需要为 custom 数据集生成每个单独的训练目标的点云数据，并将其存储在 `data/custom/custom_gt_database` 的 `.bin` 格式的文件中，此外，需要为训练数据或者验证数据生成 `.pkl` 格式的包含数据信息的文件。随后，通过运行下面的命令来创建最终的 custom 数据：

```bash
mkdir ./data/custom/ && mkdir ./data/custom/ImageSets

python tools/create_data.py custom --root-path ./data/custom --out-dir ./data/custom --extra-tag custom --with-plane
```

需要注意的是，如果您的本地磁盘没有充足的存储空间来存储转换后的数据，您可以通过改变 `--out-dir` 来指定其他任意的存储路径。如果您没有准备 `planes` 数据，您需要移除 `--with-plane` 标志。

处理后的文件夹结构应该如下：

```
custom
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   ├── val.txt
├── testing
│   ├── velodyne
├── training
│   ├── label_2
│   ├── velodyne
├── custom_gt_database
│   ├── xxxxx.bin
├── custom_infos_train.pkl
├── custom_infos_val.pkl
├── custom_dbinfos_train.pkl
├── custom_infos_test.pkl
├── custom_infos_trainval.pkl
```

- `custom_gt_database/xxxxx.bin`：训练数据集中包含在 3D 标注框中的点云数据。
- `custom_infos_train.pkl`：训练数据集，该字典包含了两个键值：`metainfo` 和 `data_list`。`metainfo` 包含数据集的基本信息，例如 `categories`, `dataset` 和 `info_version`。`data_list` 是由字典组成的列表，每个字典（以下简称 `info`）包含了单个样本的所有详细信息。
  - info\['sample_idx'\]：该样本在整个数据集的索引。
    - info\['lidar_points'\]：是一个字典，包含了激光雷达点相关的信息。
      - info\['lidar_points'\]\['lidar_path'\]：激光雷达点云数据的文件名。
      - info\['lidar_points'\]\['num_pts_feats'\]：点的特征维度。
    - info\['instances'\]：是一个字典组成的列表。每个字典包含单个实例的所有标注信息。对于其中的第 i 个实例，我们有：
      - info\['instances'\]\[i\]\['bbox'\]：长度为 4 的列表，以 (x1, y1, x2, y2) 的顺序表示实例的 2D 边界框。
      - info\['instances'\]\[i\]\['bbox_3d'\]：长度为 7 的列表，以 (x, y, z, l, h, w, yaw) 的顺序表示实例的 3D 边界框。
      - info\['instances'\]\[i\]\['bbox_label'\]：是一个整数，表示实例的 2D 标签，-1 代表忽略。
      - info\['instances'\]\[i\]\['bbox_label_3d'\]：是一个整数，表示实例的 3D 标签，-1 代表忽略。
      - info\['instances'\]\[i\]\['depth'\]：3D 边界框投影到相关图像平面的中心点的深度。
      - info\['instances'\]\[i\]\['num_lidar_pts'\]：3D 边界框内的激光雷达点数。
      - info\['instances'\]\[i\]\['center_2d'\]：3D 边界框投影的 2D 中心。
      - info\['instances'\]\[i\]\['difficulty'\]：custom 官方定义的困难度，包括简单、适中、困难。
      - info\['instances'\]\[i\]\['truncated'\]：从 0（非截断）到 1（截断）的浮点数，其中截断指的是离开检测图像边界的检测目标。
      - info\['instances'\]\[i\]\['occluded'\]：整数 (0,1,2,3) 表示目标的遮挡状态：0 = 完全可见，1 = 部分遮挡，2 = 大面积遮挡，3 = 未知。
      - info\['instances'\]\[i\]\['group_ids'\]：用于多部分的物体。