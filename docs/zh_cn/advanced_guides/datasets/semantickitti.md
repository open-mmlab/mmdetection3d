# SemanticKITTI 数据集

本页提供了有关在 MMDetection3D 中使用 SemanticKITTI 数据集的具体教程。

## 数据集准备

您可以在[这里](http://semantic-kitti.org/dataset.html#download)下载 SemanticKITTI 数据集并解压缩所有 zip 文件。

像准备数据集的一般方法一样，建议将数据集根目录链接到 `$MMDETECTION3D/data`。

在我们处理之前，文件夹结构应按如下方式组织：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 22
```

SemanticKITTI 数据集包含 23 个序列，其中序列 \[0-7\] , \[9-10\] 作为训练集（约 19k 训练样本），序列 8 作为验证集（约 4k 验证样本），\[11-22\] 作为测试集 （约20k测试样本）。其中每个序列分别包含 velodyne 和 labels 两个文件夹分别存放激光雷达点云数据和分割标注 (其中高16位存放实例分割标注，低16位存放语义分割标注)。

### 创建 SemanticKITTI 数据集

我们提供了生成数据集信息的脚本，用于测试和训练。通过以下命令生成 `.pkl` 文件：

```bash
python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti
```

处理后的文件夹结构应该如下：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 22
│   │   ├── semantickitti_infos_test.pkl
│   │   ├── semantickitti_infos_train.pkl
│   │   ├── semantickitti_infos_val.pkl
```

- `semantickitti_infos_train.pkl`: 训练数据集, 该字典包含两个键值: `metainfo` 和 `data_list`.
  `metainfo` 包含该数据集的基本信息。 `data_list` 是由字典组成的列表，每个字典（以下简称 `info`）包含了单个样本的所有详细信息。
  - info\['sample_id'\]：该样本在整个数据集的索引。
  - info\['lidar_points'\]：是一个字典，包含了激光雷达点相关的信息。
    - info\['lidar_points'\]\['lidar_path'\]：激光雷达点云数据的文件名。
    - info\['lidar_points'\]\['num_pts_feats'\]：点的特征维度
  - info\['pts_semantic_mask_pth'\]：三维语义分割的标注文件的文件路径

更多细节请参考 [semantickitti_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/semantickitti_converter.py) 和 [update_infos_to_v2.py ](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/update_infos_to_v2.py) 。

## Train pipeline

下面展示了一个使用 SemanticKITTI 数据集进行 3D 语义分割的典型流程：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
```

- 数据增强:
  - `RandomFlip3D`：对输入点云数据进行随机地水平翻转或者垂直翻转。
  - `GlobalRotScaleTrans`：对输入点云数据进行旋转、缩放、平移。

## 评估

使用 8 个 GPU 以及 SemanticKITTI 指标评估的 MinkUNet 的示例如下：

```shell
bash tools/dist_test.sh configs/minkunet/minkunet_w32_8xb2-15e_semantickitti.py checkpoints/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth 8
```

## 度量指标

通常我们使用平均交并比 (mean Intersection over Union, mIoU) 作为 SemanticKITTI 语义分割任务的度量指标。
具体而言，我们先计算所有类别的 IoU，然后取平均值作为 mIoU。
更多实现细节请参考 [seg_eval.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/evaluation/functional/seg_eval.py)。

以下是一个评估结果的样例:

| classes | car    | bicycle | motorcycle | truck  | bus    | person | bicyclist | motorcyclist | road   | parking | sidewalk | other-ground | building | fence  | vegetation | trunck | terrian | pole   | traffic-sign | miou   | acc    | acc_cls |
| ------- | ------ | ------- | ---------- | ------ | ------ | ------ | --------- | ------------ | ------ | ------- | -------- | ------------ | -------- | ------ | ---------- | ------ | ------- | ------ | ------------ | ------ | ------ | ------- |
| results | 0.9687 | 0.1908  | 0.6313     | 0.8580 | 0.6359 | 0.6818 | 0.8444    | 0.0002       | 0.9353 | 0.4854  | 0.8106   | 0.0024       | 0.9050   | 0.6111 | 0.8822     | 0.6605 | 0.7493  | 0.6442 | 0.4837       | 0.6306 | 0.9202 | 0.6924  |
