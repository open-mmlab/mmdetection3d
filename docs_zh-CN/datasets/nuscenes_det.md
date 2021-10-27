# 3D 目标检测 NuScenes 数据集

本页提供了有关在 MMDetection3D 中使用 nuScenes 数据集的具体教程。

## 准备之前

您可以在[这里](https://www.nuscenes.org/download)下载 nuScenes 3D 检测数据并解压缩所有 zip 文件。

像准备数据集的一般方法一样，建议将数据集根目录链接到 `$MMDETECTION3D/data`。

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

我们通常需要通过特定样式来使用 .pkl 或 .json 文件组织有用的数据信息，例如用于组织图像及其标注的 coco 样式。
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
接下来，我们将详细说明这些信息文件中记录的细节。

- `nuscenes_database/xxxxx.bin`：训练数据集的每个 3D 包围框中包含的点云数据。
- `nuscenes_infos_train.pkl`：训练数据集信息，每帧信息有两个键值： `metadata` 和 `infos`。 `metadata` 包含数据集本身的基本信息，例如 `{'version': 'v1.0-trainval'}`，而 `infos` 包含详细信息如下：
    - info['lidar_path']：激光雷达点云数据的文件路径。
    - info['token']：样本数据标记。
    - info['sweeps']：扫描信息（nuScenes 中的 `sweeps` 是指没有标注的中间帧，而 `samples` 是指那些带有标注的关键帧）。
        - info['sweeps'][i]['data_path']：第 i 次扫描的数据路径。
        - info['sweeps'][i]['type']：扫描数据类型，例如“激光雷达”。
        - info['sweeps'][i]['sample_data_token']：扫描样本数据标记。
        - info['sweeps'][i]['sensor2ego_translation']：从当前传感器（用于收集扫描数据）到自车（包含感知周围环境传感器的车辆，车辆坐标系固连在自车上）的转换（1x3 列表）。
        - info['sweeps'][i]['sensor2ego_rotation']：从当前传感器（用于收集扫描数据）到自车的旋转（四元数格式的 1x4 列表）。
        - info['sweeps'][i]['ego2global_translation']：从自车到全局坐标的转换（1x3 列表）。
        - info['sweeps'][i]['ego2global_rotation']：从自车到全局坐标的旋转（四元数格式的 1x4 列表）。
        - info['sweeps'][i]['timestamp']：扫描数据的时间戳。
        - info['sweeps'][i]['sensor2lidar_translation']：从当前传感器（用于收集扫描数据）到激光雷达的转换（1x3 列表）。
        - info['sweeps'][i]['sensor2lidar_rotation']：从当前传感器（用于收集扫描数据）到激光雷达的旋转（四元数格式的 1x4 列表）。
    - info['cams']：相机校准信息。它包含与每个摄像头对应的六个键值： `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`。
    每个字典包含每个扫描数据按照上述方式的详细信息（每个信息的关键字与上述相同）。
    - info['lidar2ego_translation']：从激光雷达到自车的转换（1x3 列表）。
    - info['lidar2ego_rotation']：从激光雷达到自车的旋转（四元数格式的 1x4 列表）。
    - info['ego2global_translation']：从自车到全局坐标的转换（1x3 列表）。
    - info['ego2global_rotation']：从自我车辆到全局坐标的旋转（四元数格式的 1x4 列表）。
    - info['timestamp']：样本数据的时间戳。
    - info['gt_boxes']：7 个自由度的 3D 包围框，一个 Nx7 数组。
    - info['gt_names']：3D 包围框的类别，一个 1xN 数组。
    - info['gt_velocity']：3D 包围框的速度（由于不准确，没有垂直测量），一个 Nx2 数组。
    - info['num_lidar_pts']：每个 3D 包围框中包含的激光雷达点数。
    - info['num_radar_pts']：每个 3D 包围框中包含的雷达点数。
    - info['valid_flag']：每个包围框是否有效。一般情况下，我们只将包含至少一个激光雷达或雷达点的 3D 框作为有效框。
- `nuscenes_infos_train_mono3d.coco.json`：训练数据集 coco 风格的信息。该文件将基于图像的数据组织为三类（键值）：`'categories'`, `'images'`, `'annotations'`。
    - info['categories']：包含所有类别名称的列表。每个元素都遵循字典格式并由两个键值组成：`'id'` 和 `'name'`。
    - info['images']：包含所有图像信息的列表。
        - info['images'][i]['file_name']：第 i 张图像的文件名。
        - info['images'][i]['id']：第 i 张图像的样本数据标记。
        - info['images'][i]['token']：与该帧对应的样本标记。
        - info['images'][i]['cam2ego_rotation']：从相机到自车的旋转（四元数格式的 1x4 列表）。
        - info['images'][i]['cam2ego_translation']：从相机到自车的转换（1x3 列表）。
        - info['images'][i]['ego2global_rotation'']：从自车到全局坐标的旋转（四元数格式的 1x4 列表）。
        - info['images'][i]['ego2global_translation']：从自车到全局坐标的转换（1x3 列表）。
        - info['images'][i]['cam_intrinsic']: 相机内参矩阵（3x3 列表）。
        - info['images'][i]['width']：图片宽度， nuScenes 中默认为 1600。
        - info['images'][i]['height']：图像高度， nuScenes 中默认为 900。
    - info['annotations']: 包含所有标注信息的列表。
        - info['annotations'][i]['file_name']：对应图像的文件名。
        - info['annotations'][i]['image_id']：对应图像的图像 ID （标记）。
        - info['annotations'][i]['area']：2D 包围框的面积。
        - info['annotations'][i]['category_name']：类别名称。
        - info['annotations'][i]['category_id']：类别 id。
        - info['annotations'][i]['bbox']：2D 包围框标注（3D 投影框的外部矩形），1x4 列表跟随 [x1, y1, x2-x1, y2-y1]。x1/y1 是沿图像水平/垂直方向的最小坐标。
        - info['annotations'][i]['iscrowd']：该区域是否拥挤。默认为 0。
        - info['annotations'][i]['bbox_cam3d']：3D 包围框（重力）中心位置（3）、大小（3）、（全局）偏航角（1）、1x7 列表。
        - info['annotations'][i]['velo_cam3d']：3D 包围框的速度（由于不准确，没有垂直测量），一个 Nx2 数组。
        - info['annotations'][i]['center2d']：包含 2.5D 信息的投影 3D 中心：图像上的投影中心位置（2）和深度（1），1x3 列表。
        - info['annotations'][i]['attribute_name']：属性名称。
        - info['annotations'][i]['attribute_id']：属性 ID。
        我们为属性分类维护了一个属性集合和映射。更多的细节请参考[这里](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L53)。
        - info['annotations'][i]['id']：标注 ID。默认为 `i`。

这里我们只解释训练信息文件中记录的数据。这同样适用于验证和测试集。
获取 `nuscenes_infos_xxx.pkl` 和 `nuscenes_infos_xxx_mono3d.coco.json` 的核心函数分别为 [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py#L143) 和 [get_2d_boxes](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py#L397)。更多细节请参考 [nuscenes_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py)。

## 训练流程

### 基于 LiDAR 的方法

nuScenes 上基于 LiDAR 的 3D 检测（包括多模态方法）的典型训练流程如下。

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
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

与一般情况相比，nuScenes 有一个特定的 `'LoadPointsFromMultiSweeps'` 流水线来从连续帧加载点云。这是此设置中使用的常见做法。
更多细节请参考 nuScenes [原始论文](https://arxiv.org/abs/1903.11027)。
`'LoadPointsFromMultiSweeps'` 中的默认 `use_dim` 是 `[0, 1, 2, 4]`，其中前 3 个维度是指点坐标，最后一个是指时间戳差异。
由于在拼接来自不同帧的点时使用点云的强度信息会产生噪声，因此默认情况下不使用点云的强度信息。

### 基于视觉的方法

nuScenes 上基于图像的 3D 检测的典型训练流水线如下。

```python
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
```

它遵循 2D 检测的一般流水线，但在一些细节上有所不同：
- 它使用单目流水线加载图像，其中包括额外的必需信息，如相机内参矩阵。
- 它需要加载 3D 标注。
- 一些数据增强技术需要调整，例如`RandomFlip3D`。
目前我们不支持更多的增强方法，因为如何迁移和应用其他技术仍在探索中。

## 评估

使用 8 个 GPU 以及 nuScenes 指标评估的 PointPillars 的示例如下

```shell
bash ./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth 8 --eval bbox
```

## 指标

NuScenes 提出了一个综合指标，即 nuScenes 检测分数（NDS），以评估不同的方法并设置基准测试。
它由平均精度（mAP）、平均平移误差（ATE）、平均尺度误差（ASE）、平均方向误差（AOE）、平均速度误差（AVE）和平均属性误差（AAE）组成。
更多细节请参考其[官方网站](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)。

我们也采用这种方法对 nuScenes 进行评估。打印的评估结果示例如下：

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.503   0.577   0.152   0.111   2.096   0.136
truck   0.223   0.857   0.224   0.220   1.389   0.179
bus     0.294   0.855   0.204   0.190   2.689   0.283
trailer 0.081   1.094   0.243   0.553   0.742   0.167
construction_vehicle    0.058   1.017   0.450   1.019   0.137   0.341
pedestrian      0.392   0.687   0.284   0.694   0.876   0.158
motorcycle      0.317   0.737   0.265   0.580   2.033   0.104
bicycle 0.308   0.704   0.299   0.892   0.683   0.010
traffic_cone    0.555   0.486   0.309   nan     nan     nan
barrier 0.466   0.581   0.269   0.169   nan     nan
```

## 测试和提交

使用 8 个 GPU 在 nuScenes 上测试 PointPillars 并生成对排行榜的提交的示例如下

```shell
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth 8 --out work_dirs/pp-nus/results_eval.pkl --format-only --eval-options 'jsonfile_prefix=work_dirs/pp-nus/results_eval'
```

请注意，在[这里](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/nus-3d.py#L132)测试信息应更改为测试集而不是验证集。

生成 `work_dirs/pp-nus/results_eval.json` 后，您可以压缩并提交给 nuScenes 基准测试。更多信息请参考 [nuScenes 官方网站](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)。

我们还可以使用我们开发的可视化工具将预测结果可视化。更多细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#id2)。

## 注意

### `NuScenesBox` 和我们的 `CameraInstanceBoxes` 之间的转换。

总的来说，`NuScenesBox` 和我们的 `CameraInstanceBoxes` 的主要区别主要体现在转向角（yaw）定义上。 `NuScenesBox` 定义了一个四元数或三个欧拉角的旋转，而我们的由于实际情况只定义了一个转向角（yaw），它需要我们在预处理和后处理中手动添加一些额外的旋转，例如[这里](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L673)。

另外，请注意，角点和位置的定义在 `NuScenesBox` 中是分离的。例如，在单目 3D 检测中，框位置的定义在其相机坐标中（有关汽车设置，请参阅其官方[插图](https://www.nuscenes.org/nuscenes#data-collection)），即与[我们的](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py)一致。相比之下，它的角点是通过[惯例](https://github.com/nutonomy/nuscenes-devkit/blob/02e9200218977193a1058dd7234f935834378319/python-sdk/nuscenes/utils/data_classes.py#L527) 定义的，“x 向前， y 向左， z 向上”。它导致了与我们的 `CameraInstanceBoxes` 不同的维度和旋转定义理念。一个移除相似冲突的例子是 PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744)。同样的问题也存在于 LiDAR 系统中。为了解决它们，我们通常会在预处理和后处理中添加一些转换，以保证在整个训练和推理过程中框都在我们的坐标系系统里。
