# NuScenes 数据集

本页提供了有关在 MMDetection3D 中使用 nuScenes 数据集的具体教程。

## 准备之前

您可以在[这里](https://www.nuscenes.org/download)下载 nuScenes 3D 检测数据 Full dataset (v1.0) 并解压缩所有 zip 文件。

如果您想进行 3D 语义分割任务，需要额外下载 nuScenes-lidarseg 数据标注，并将解压的文件放入 nuScenes 对应的文件夹下。

**注意**：nuScenes-lidarseg 中的 v1.0trainval(test)/categroy.json 会替换原先 Full dataset (v1.0) 原先的 v1.0trainval(test)/categroy.json，但是不会对 3D 目标检测任务造成影响。

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
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## 数据准备

我们通常需要通过特定样式来使用 `.pkl` 文件组织有用的数据信息。要为 nuScenes 准备这些文件，请运行以下命令：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

处理后的文件夹结构应该如下。

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
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

- `nuscenes_database/xxxxx.bin`：训练数据集的每个 3D 包围框中包含的点云数据。
- `nuscenes_infos_train.pkl`：训练数据集，该字典包含了两个键值：`metainfo` 和 `data_list`。`metainfo` 包含数据集的基本信息，例如 `categories`, `dataset` 和 `info_version`。`data_list` 是由字典组成的列表，每个字典（以下简称 `info`）包含了单个样本的所有详细信息。
  - info\['sample_idx'\]：样本在整个数据集的索引。
  - info\['token'\]：样本数据标记。
  - info\['timestamp'\]：样本数据时间戳。
  - info\['ego2global'\]：自车到全局坐标的变换矩阵。（4x4 列表）
  - info\['lidar_points'\]：是一个字典，包含了所有与激光雷达点相关的信息。
    - info\['lidar_points'\]\['lidar_path'\]：激光雷达点云数据的文件名。
    - info\['lidar_points'\]\['num_pts_feats'\]：点的特征维度。
    - info\['lidar_points'\]\['lidar2ego'\]：该激光雷达传感器到自车的变换矩阵。（4x4 列表）
  - info\['lidar_sweeps'\]：是一个列表，包含了扫描信息（没有标注的中间帧）。
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['data_path'\]：第 i 次扫描的激光雷达数据的文件路径。
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\[lidar2ego''\]：当前激光雷达传感器到自车的变换矩阵。（4x4 列表）
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['ego2global'\]：自车到全局坐标的变换矩阵。（4x4 列表）
    - info\['lidar_sweeps'\]\[i\]\['lidar2sensor'\]：从主激光雷达传感器到当前传感器（用于收集扫描数据）的变换矩阵。（4x4 列表）
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
    - info\['instances'\]\[i\]\['bbox_3d'\]：长度为 7 的列表，以 (x, y, z, l, w, h, yaw) 的顺序表示实例的 3D 边界框。
    - info\['instances'\]\[i\]\['bbox_label_3d'\]：整数表示实例的标签，-1 代表忽略。
    - info\['instances'\]\[i\]\['velocity'\]：3D 边界框的速度（由于不正确，没有垂直测量），大小为 (2, ) 的列表。
    - info\['instances'\]\[i\]\['num_lidar_pts'\]：每个 3D 边界框内包含的激光雷达点数。
    - info\['instances'\]\[i\]\['num_radar_pts'\]：每个 3D 边界框内包含的雷达点数。
    - info\['instances'\]\[i\]\['bbox_3d_isvalid'\]：每个包围框是否有效。一般情况下，我们只将包含至少一个激光雷达或雷达点的 3D 框作为有效框。
  - info\['cam_instances'\]：是一个字典，包含以下键值：`'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`。对于基于视觉的 3D 目标检测任务，我们将整个场景的 3D 标注划分至它们所属于的相应相机中。对于其中的第 i 个实例，我们有：
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label'\]：实例标签。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label_3d'\]：实例标签。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox'\]：2D 边界框标注（3D 框投影的矩形框），顺序为 \[x1, y1, x2, y2\] 的列表。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['center_2d'\]：3D 框投影到图像上的中心点，大小为 (2, ) 的列表。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['depth'\]：3D 框投影中心的深度。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['velocity'\]：3D 边界框的速度（由于不正确，没有垂直测量），大小为 (2, ) 的列表。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['attr_label'\]：实例的属性标签。我们为属性分类维护了一个属性集合和映射。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_3d'\]：长度为 7 的列表，以 (x, y, z, l, h, w, yaw) 的顺序表示实例的 3D 边界框。
  - info\['pts_semantic_mask_path'\]：激光雷达语义分割标注的文件名。

注意：

1. `instances` 和 `cam_instances` 中 `bbox_3d` 的区别。`bbox_3d` 都被转换到 MMDet3D 定义的坐标系下，`instances` 中的 `bbox_3d` 是在激光雷达坐标系下，而 `cam_instances` 是在相机坐标系下。注意它们 3D 框中表示的不同（'l, w, h' 和 'l, h, w'）。

2. 这里我们只解释训练信息文件中记录的数据。这同样适用于验证集和测试集（测试集的 `.pkl` 文件中不包含 `instances` 以及 `cam_instances`）。

获取 `nuscenes_infos_xxx.pkl` 的核心函数为 [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/nuscenes_converter.py#L146)。更多细节请参考 [nuscenes_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/nuscenes_converter.py)。

## 训练流程

### 基于 LiDAR 的方法

nuScenes 上基于 LiDAR 的 3D 检测（包括多模态方法）的典型训练流程如下。

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
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

与一般情况相比，nuScenes 有一个特定的 `'LoadPointsFromMultiSweeps'` 流水线来从连续帧加载点云。这是此设置中使用的常见做法。更多细节请参考 nuScenes [原始论文](https://arxiv.org/abs/1903.11027)。`'LoadPointsFromMultiSweeps'` 中的默认 `use_dim` 是 `[0, 1, 2, 4]`，其中前 3 个维度是指点坐标，最后一个是指时间戳差异。由于在拼接来自不同帧的点时使用点云的强度信息会产生噪声，因此默认情况下不使用点云的强度信息。

### 基于视觉的方法

#### 基于单目方法

在NuScenes数据集中，对于多视角图像，单目检测范式通常由针对每张图像检测和输出 3D 检测结果以及通过后处理（例如 NMS ）得到最终检测结果两步组成。从本质上来说，这种范式直接将单目 3D 检测扩展到多视角任务。NuScenes 上基于图像的 3D 检测的典型训练流水线如下。

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
    dict(type='mmdet.Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
```

它遵循 2D 检测的一般流水线，但在一些细节上有所不同：

- 它使用单目流水线加载图像，其中包括额外的必需信息，如相机内参矩阵。
- 它需要加载 3D 标注。
- 一些数据增强技术需要调整，例如`RandomFlip3D`。目前我们不支持更多的增强方法，因为如何迁移和应用其他技术仍在探索中。

#### 基于BEV方法

鸟瞰图，BEV（Bird's-Eye-View），是另一种常用的 3D 检测范式。它直接利用多个视角图像进行 3D 检测。对于 NuScenes 数据集而言，这些视角包括前方`CAM_FRONT`、左前方`CAM_FRONT_LEFT`、右前方`CAM_FRONT_RIGHT`、后方`CAM_BACK`、左后方`CAM_BACK_LEFT`、右后方`CAM_BACK_RIGHT`。一个基本的用于 BEV 方法的流水线如下。

```python
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
train_transforms = [
    dict(type='PhotoMetricDistortion3D'),
    dict(
        type='RandomResize3D',
        scale=(1600, 900),
        ratio_range=(1., 1.),
        keep_ratio=True)
]
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=6, ),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    # 可选，数据增强
    dict(type='MultiViewWrapper', transforms=train_transforms),
    # 可选, 筛选特定点云范围内物体
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # 可选, 筛选特定类别物体
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

为了读取多个视角的图像，数据集也应进行相应微调。

```python
data_prefix = dict(
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
)
train_dataloader = dict(
    batch_size=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type="NuScenesDataset",
        data_root="./data/nuScenes",
        ann_file="nuscenes_infos_train.pkl",
        data_prefix=data_prefix,
        modality=dict(use_camera=True, use_lidar=False, ),
        pipeline=train_pipeline,
        test_mode=False, )
)
```

## 评估

使用 8 个 GPU 以及 nuScenes 指标评估的 PointPillars 的示例如下

```shell
bash ./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth 8
```

## 指标

NuScenes 提出了一个综合指标，即 nuScenes 检测分数（NDS），以评估不同的方法并设置基准测试。它由平均精度（mAP）、平均平移误差（ATE）、平均尺度误差（ASE）、平均方向误差（AOE）、平均速度误差（AVE）和平均属性误差（AAE）组成。更多细节请参考其[官方网站](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)。

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

你需要在对应的配置文件中的 `test_evaluator` 里修改 `jsonfile_prefix`。举个例子，添加 `test_evaluator = dict(type='NuScenesMetric', jsonfile_prefix='work_dirs/pp-nus/results_eval.json')` 或在测试命令后使用 `--cfg-options "test_evaluator.jsonfile_prefix=work_dirs/pp-nus/results_eval.json)`。

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py work_dirs/pp-nus/latest.pth 8 --cfg-options 'test_evaluator.jsonfile_prefix=work_dirs/pp-nus/results_eval'
```

请注意，在[这里](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/nus-3d.py#L132)测试信息应更改为测试集而不是验证集。

生成 `work_dirs/pp-nus/results_eval.json` 后，您可以压缩并提交给 nuScenes 基准测试。更多信息请参考 [nuScenes 官方网站](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)。

我们还可以使用我们开发的可视化工具将预测结果可视化。更多细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#id2)。

## 注意

### `NuScenesBox` 和我们的 `CameraInstanceBoxes` 之间的转换。

总的来说，`NuScenesBox` 和我们的 `CameraInstanceBoxes` 的主要区别主要体现在转向角（yaw）定义上。 `NuScenesBox` 定义了一个四元数或三个欧拉角的旋转，而我们的由于实际情况只定义了一个转向角（yaw），它需要我们在预处理和后处理中手动添加一些额外的旋转，例如[这里](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L673)。

另外，请注意，角点和位置的定义在 `NuScenesBox` 中是分离的。例如，在单目 3D 检测中，框位置的定义在其相机坐标中（有关汽车设置，请参阅其官方[插图](https://www.nuscenes.org/nuscenes#data-collection)），即与[我们的](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py)一致。相比之下，它的角点是通过[惯例](https://github.com/nutonomy/nuscenes-devkit/blob/02e9200218977193a1058dd7234f935834378319/python-sdk/nuscenes/utils/data_classes.py#L527) 定义的，“x 向前， y 向左， z 向上”。它导致了与我们的 `CameraInstanceBoxes` 不同的维度和旋转定义理念。一个移除相似冲突的例子是 PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744)。同样的问题也存在于 LiDAR 系统中。为了解决它们，我们通常会在预处理和后处理中添加一些转换，以保证在整个训练和推理过程中框都在我们的坐标系系统里。
