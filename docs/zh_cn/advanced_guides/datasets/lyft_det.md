# 3D 目标检测 Lyft 数据集

本页提供了有关在 MMDetection3D 中使用 Lyft 数据集的具体教程。

## 准备之前

您可以在[这里](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data)下载 Lyft 3D 检测数据并解压缩所有 zip 文件。

像准备数据集的一般方法一样，建议将数据集根目录链接到 `$MMDETECTION3D/data`。

在进行处理之前，文件夹结构应按如下方式组织：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (train_data)
│   │   │   ├── lidar (train_lidar)
│   │   │   ├── images (train_images)
│   │   │   ├── maps (train_maps)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (test_data)
│   │   │   ├── lidar (test_lidar)
│   │   │   ├── images (test_images)
│   │   │   ├── maps (test_maps)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
```

其中 `v1.01-train` 和 `v1.01-test` 包含与 nuScenes 数据集相同的元文件，`.txt` 文件包含数据划分的信息。
Lyft 不提供训练集和验证集的官方划分方案，因此 MMDetection3D 对不同场景下的不同类别的目标数量进行分析，并提供了一个数据集划分方案。
`sample_submission.csv` 是用于提交到 Kaggle 评估服务器的基本文件。
需要注意的是，我们遵循了 Lyft 最初的文件夹命名以实现更清楚的文件组织。请将下载下来的原始文件夹重命名按照上述组织结构重新命名。

## 数据准备

组织 Lyft 数据集的方式和组织 nuScenes 的方式相同，首先会生成几乎具有相同结构的 .pkl 和 .json 文件，接着需要重点关注这两个数据集之间的不同点，请参考 [nuScenes 教程](https://github.com/open-mmlab/mmdetection3d/blob/master/docs_zh-CN/datasets/nuscenes_det.md)获取更加详细的数据集信息文件结构的说明。

请通过运行下面的命令来生成 Lyft 的数据集信息文件：

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/data_converter/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

请注意，上面的第二行命令用于修复损坏的 lidar 数据文件，请参考[此处](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000)获取更多细节。

处理后的文件夹结构应该如下：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (train_data)
│   │   │   ├── lidar (train_lidar)
│   │   │   ├── images (train_images)
│   │   │   ├── maps (train_maps)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (test_data)
│   │   │   ├── lidar (test_lidar)
│   │   │   ├── images (test_images)
│   │   │   ├── maps (test_maps)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
│   │   ├── lyft_infos_train.pkl
│   │   ├── lyft_infos_val.pkl
│   │   ├── lyft_infos_test.pkl
│   │   ├── lyft_infos_train_mono3d.coco.json
│   │   ├── lyft_infos_val_mono3d.coco.json
│   │   ├── lyft_infos_test_mono3d.coco.json
```

其中，.pkl 文件通常适用于涉及到点云的相关方法，coco 类型的 .json 文件更加适用于涉及到基于图像的相关方法，如基于图像的 2D 和 3D 目标检测。
不同于 nuScenes 数据集，这里仅能使用 json 文件进行 2D 检测相关的实验，未来将会进一步支持基于图像的 3D 检测。

接下来将详细介绍 Lyft 数据集和 nuScenes 数据集之间的数据集信息文件中的不同点：

- `lyft_database/xxxxx.bin` 文件不存在：由于真实标注框的采样对实验的影响可以忽略不计，在 Lyft 数据集中不会提取该目录和相关的 `.bin` 文件。
- `lyft_infos_train.pkl`：包含训练数据集信息，每一帧包含两个关键字：`metadata` 和 `infos`。
  `metadata` 包含数据集自身的基础信息，如 `{'version': 'v1.01-train'}`，然而 `infos` 包含和 nuScenes 数据集相似的数据集详细信息，但是并不包含一下几点：
  - info\['sweeps'\]：扫描信息.
    - info\['sweeps'\]\[i\]\['type'\]：扫描信息的数据类型，如 `'lidar'`。
      Lyft 数据集中的一些样例具有不同的 LiDAR 设置，然而为了数据分布的一致性，这里将一直采用顶部的 LiDAR 设备所采集的数据点信息。
  - info\['gt_names'\]：在 Lyft 数据集中有 9 个类别，相比于 nuScenes 数据集，不同类别的标注不平衡问题更加突出。
  - info\['gt_velocity'\] 不存在：Lyft 数据集中不存在速度评估信息。
  - info\['num_lidar_pts'\]：默认值设置为 -1。
  - info\['num_radar_pts'\]：默认值设置为 0。
  - info\['valid_flag'\] 不存在：这个标志信息因无效的 `num_lidar_pts` 和 `num_radar_pts` 的存在而存在。
- `nuscenes_infos_train_mono3d.coco.json`：包含 coco 类型的训练数据集相关的信息。这个文件仅包含 2D 相关的信息，不包含 3D 目标检测所需要的信息，如相机内参。
  - info\['images'\]：包含所有图像信息的列表。
    - 仅包含 `'file_name'`, `'id'`, `'width'`, `'height'`。
  - info\['annotations'\]：包含所有标注信息的列表。
    - 仅包含 `'file_name'`，`'image_id'`，`'area'`，`'category_name'`，`'category_id'`，`'bbox'`，`'is_crowd'`，`'segmentation'`，`'id'`，其中 `'is_crowd'` 和 `'segmentation'` 默认设置为 `0` 和 `[]`。
      Lyft 数据集中不包含属性标注信息。

这里仅介绍存储在训练数据文件的数据记录信息，在测试数据集也采用上述的数据记录方式。

获取 `lyft_infos_xxx.pkl` 的核心函数是 [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/lyft_converter.py#L93)。
请参考 [lyft_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/lyft_converter.py) 获取更多细节。

## 训练流程

### 基于 LiDAR 的方法

Lyft 上基于 LiDAR 的 3D 检测（包括多模态方法）的训练流程与 nuScenes 几乎相同，如下所示：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

与 nuScenes 相似，在 Lyft 上进行训练的模型也需要 `LoadPointsFromMultiSweeps` 步骤来从连续帧中加载点云数据。
另外，考虑到 Lyft 中所收集的激光雷达点的强度是无效的，因此将 `LoadPointsFromMultiSweeps` 中的 `use_dim` 默认值设置为 `[0, 1, 2, 4]`，其中前三个维度表示点的坐标，最后一个维度表示时间戳的差异。

## 评估

使用 8 个 GPU 以及 Lyft 指标评估的 PointPillars 的示例如下：

```shell
bash ./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth 8 --eval bbox
```

## 度量指标

Lyft 提出了一个更加严格的用以评估所预测的 3D 检测框的度量指标。
判断一个预测框是否是正类的基本评判标准和 KITTI 一样，如基于 3D 交并比进行评估，然而，Lyft 采用与 COCO 相似的方式来计算平均精度 -- 计算 3D 交并比在 0.5-0.95 之间的不同阈值下的平均精度。
实际上，重叠部分大于 0.7 的 3D 交并比是一项对于 3D 检测方法比较严格的标准，因此整体的性能似乎会偏低。
相比于其他数据集，Lyft 上不同类别的标注不平衡是导致最终结果偏低的另一个重要原因。
请参考[官方网址](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview/evaluation)获取更多关于度量指标的定义的细节。

这里将采用官方方法对 Lyft 进行评估，下面展示了一个评估结果的例子：

```
+mAPs@0.5:0.95------+--------------+
| class             | mAP@0.5:0.95 |
+-------------------+--------------+
| animal            | 0.0          |
| bicycle           | 0.099        |
| bus               | 0.177        |
| car               | 0.422        |
| emergency_vehicle | 0.0          |
| motorcycle        | 0.049        |
| other_vehicle     | 0.359        |
| pedestrian        | 0.066        |
| truck             | 0.176        |
| Overall           | 0.15         |
+-------------------+--------------+
```

## 测试和提交

使用 8 个 GPU 在 Lyft 上测试 PointPillars 并生成对排行榜的提交的示例如下：

```shell
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py work_dirs/pp-lyft/latest.pth 8 --out work_dirs/pp-lyft/results_challenge.pkl --format-only --eval-options 'jsonfile_prefix=work_dirs/pp-lyft/results_challenge' 'csv_savepath=results/pp-lyft/results_challenge.csv'
```

在生成 `work_dirs/pp-lyft/results_challenge.csv`，您可以将生成的文件提交到 Kaggle 评估服务器，请参考[官方网址](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)获取更多细节。

同时还可以使用可视化工具将预测结果进行可视化，请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#visualization)获取更多细节。
