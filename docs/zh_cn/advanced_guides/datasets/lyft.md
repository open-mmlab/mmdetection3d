# Lyft 数据集

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

其中 `v1.01-train` 和 `v1.01-test` 包含与 nuScenes 数据集相同的元文件，`.txt` 文件包含数据划分的信息。Lyft 不提供训练集和验证集的官方划分方案，因此 MMDetection3D 对不同场景下的不同类别的目标数量进行分析，并提供了一个数据集划分方案。`sample_submission.csv` 是用于提交到 Kaggle 评估服务器的基本文件。需要注意的是，我们遵循了 Lyft 最初的文件夹命名以实现更清楚的文件组织。请将下载下来的原始文件夹按照上述组织结构重新命名。

## 数据准备

组织 Lyft 数据集的方式和组织 nuScenes 的方式相同，首先会生成几乎具有相同结构的 `.pkl` 文件，接着需要重点关注这两个数据集之间的不同点，更多关于数据集信息文件结构的说明请参考 [nuScenes 教程](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/zh_cn/advanced_guides/datasets/nuscenes_det.md)。

请通过运行下面的命令来生成 Lyft 的数据集信息文件：

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/data_converter/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

请注意，上面的第二行命令用于修复损坏的 lidar 数据文件，更多细节请参考此处[讨论](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000)。

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
```

- `lyft_infos_train.pkl`：训练数据集信息，该字典包含两个关键字：`metainfo` 和 `data_list`。`metainfo` 包含数据集的基本信息，例如 `categories`, `dataset` 和 `info_version`。`data_list` 是由字典组成的列表，每个字典（以下简称 `info`）包含了单个样本的所有详细信息。
  - info\['sample_idx'\]：样本在整个数据集的索引。
  - info\['token'\]：样本数据标记。
  - info\['timestamp'\]：样本数据时间戳。
  - info\['lidar_points'\]：是一个字典，包含了所有与激光雷达点相关的信息。
    - info\['lidar_points'\]\['lidar_path'\]：激光雷达点云数据的文件名。
    - info\['lidar_points'\]\['num_pts_feats'\]：点的特征维度。
    - info\['lidar_points'\]\['lidar2ego'\]：该激光雷达传感器到自车的变换矩阵。（4x4 列表）
    - info\['lidar_points'\]\['ego2global'\]：自车到全局坐标的变换矩阵。（4x4 列表）
  - info\['lidar_sweeps'\]：是一个列表，包含了扫描信息（没有标注的中间帧）。
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['data_path'\]：第 i 次扫描的激光雷达数据的文件路径。
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\[lidar2ego''\]：当前激光雷达传感器到自车在第 i 次扫描的变换矩阵。（4x4 列表）
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['ego2global'\]：自车在第 i 次扫描到全局坐标的变换矩阵。（4x4 列表）
    - info\['lidar_sweeps'\]\[i\]\['lidar2sensor'\]：从当前帧主激光雷达到第 i 帧扫描激光雷达的变换矩阵。（4x4 列表）
    - info\['lidar_sweeps'\]\[i\]\['timestamp'\]：扫描数据的时间戳。
    - info\['lidar_sweeps'\]\[i\]\['sample_data_token'\]：扫描样本数据标记。
  - info\['images'\]：是一个字典，包含与每个相机对应的六个键值：`'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`。每个字典包含了对应相机的所有数据信息。
    - info\['images'\]\['CAM_XXX'\]\['img_path'\]：图像的文件名。
    - info\['images'\]\['CAM_XXX'\]\['cam2img'\]：当 3D 点投影到图像平面时需要的内参信息相关的变换矩阵。（3x3 列表）
    - info\['images'\]\['CAM_XXX'\]\['sample_data_token'\]：图像样本数据标记。
    - info\['images'\]\['CAM_XXX'\]\['timestamp'\]：图像的时间戳。
    - info\['images'\]\['CAM_XXX'\]\['cam2ego'\]：该相机传感器到自车的变换矩阵。（4x4 列表）
    - info\['images'\]\['CAM_XXX'\]\['lidar2cam'\]：激光雷达传感器到该相机的变换矩阵。（4x4 列表）
  - info\['instances'\]：是一个字典组成的列表。每个字典包含单个实例的所有标注信息。对于其中的第 i 个实例，我们有：
    - info\['instances'\]\[i\]\['bbox_3d'\]：长度为 7 的列表，以 (x, y, z, l, w, h, yaw) 的顺序表示实例在激光雷达坐标系下的 3D 边界框。
    - info\['instances'\]\[i\]\['bbox_label_3d'\]：整数从 0 开始表示实例的标签，其中 -1 代表忽略该类别。
    - info\['instances'\]\[i\]\['bbox_3d_isvalid'\]：每个包围框是否有效。一般情况下，我们只将包含至少一个激光雷达或雷达点的 3D 框作为有效框。

接下来将详细介绍 Lyft 数据集和 nuScenes 数据集之间的数据集信息文件中的不同点：

- `lyft_database/xxxxx.bin` 文件不存在：由于真实标注框的采样对实验的影响可以忽略不计，在 Lyft 数据集中不会提取该目录和相关的 `.bin` 文件。

- `lyft_infos_train.pkl`

  - info\['instances'\]\[i\]\['velocity'\] 不存在：Lyft 数据集中不存在速度评估信息。
  - info\['instances'\]\[i\]\['num_lidar_pts'\] 及 info\['instances'\]\[i\]\['num_radar_pts'\] 不存在。

这里仅介绍存储在训练数据文件的数据记录信息。这同样适用于验证集和测试集（没有实例）。

更多关于 `lyft_infos_xxx.pkl` 的结构信息请参考 [lyft_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/lyft_converter.py)。

## 训练流程

### 基于 LiDAR 的方法

Lyft 上基于 LiDAR 的 3D 检测（包括多模态方法）的训练流程与 nuScenes 几乎相同，如下所示：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

与 nuScenes 相似，在 Lyft 上进行训练的模型也需要 `LoadPointsFromMultiSweeps` 步骤来从连续帧中加载点云数据。另外，考虑到 Lyft 中所收集的激光雷达点的强度是无效的，因此将 `LoadPointsFromMultiSweeps` 中的 `use_dim` 默认值设置为 `[0, 1, 2, 4]`，其中前三个维度表示点的坐标，最后一个维度表示时间戳的差异。

## 评估

使用 8 个 GPU 以及 Lyft 指标评估的 PointPillars 的示例如下：

```shell
bash ./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth 8
```

## 度量指标

Lyft 提出了一个更加严格的用以评估所预测的 3D 检测框的度量指标。判断一个预测框是否是正类的基本评判标准和 KITTI 一样，如基于 3D 交并比进行评估，然而，Lyft 采用与 COCO 相似的方式来计算平均精度 -- 计算 3D 交并比在 0.5-0.95 之间的不同阈值下的平均精度。实际上，重叠部分大于 0.7 的 3D 交并比是一项对于 3D 检测方法比较严格的标准，因此整体的性能似乎会偏低。相比于其他数据集，Lyft 上不同类别的标注不平衡是导致最终结果偏低的另一个重要原因。更多关于度量指标的定义请参考[官方网址](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview/evaluation)。

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
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py work_dirs/pp-lyft/latest.pth 8 --cfg-options test_evaluator.jsonfile_prefix=work_dirs/pp-lyft/results_challenge  test_evaluator.csv_savepath=results/pp-lyft/results_challenge.csv
```

在生成 `work_dirs/pp-lyft/results_challenge.csv`，您可以将生成的文件提交到 Kaggle 评估服务器，请参考[官方网址](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)获取更多细节。

同时还可以使用可视化工具将预测结果进行可视化，更多细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#visualization)。
