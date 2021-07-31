# 3D 语义分割 ScanNet 数据集

## 数据集的准备

ScanNet 3D 语义分割数据集的准备和 3D 检测任务的准备很相似，请查看[此文档](https://github.com/open-mmlab/mmdetection3d/blob/master/docs_zh-CN/datasets/scannet_det.md#dataset-preparation)以获取更多细节。
以下我们只罗列部分 3D 语义分割特有的处理步骤和数据信息。

### 提取 ScanNet 数据

因为 ScanNet 测试集对 3D 语义分割任务提供在线评测的基准，我们也需要下载其测试集并置于 `scannet` 目录下。
数据预处理前的文件目录结构应如下所示：


```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── scannet
│   │   ├── meta_data
│   │   ├── scans
│   │   │   ├── scenexxxx_xx
│   │   ├── scans_test
│   │   │   ├── scenexxxx_xx
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
```

在 `scans_test` 目录下有 100 个测试集 scan 的文件夹，每个文件夹仅包含了原始点云数据和基础的数据元文件。
例如，在 `scene0707_00` 这一目录下的文件如下所示：

- `scene0707_00_vh_clean_2.ply`：原始网格文件，存储有每个顶点的坐标和颜色。网格的顶点会被选取作为处理后点云中的点。
- `scene0707_00.txt`：数据的元文件，包含数据采集传感器的参数等信息。注意，与 `scans` 目录下的数据 (训练集和验证集) 不同，测试集 scan 并没有提供用于和坐标轴对齐的变换矩阵 (`axis-aligned matrix`)。

用户可以通过运行 `python batch_load_scannet_data.py` 指令来从原始文件中提取 ScanNet 数据。
注意，测试集只会保存下点云数据，因为没有提供标注信息。

### 创建数据集

与 3D 检测任务类似，我们通过运行 `python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet` 指令即可创建 ScanNet 数据集。
预处理后的数据目录结构如下所示：

```
scannet
├── scannet_utils.py
├── batch_load_scannet_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── README.md
├── scans
├── scans_test
├── scannet_instance_data
├── points
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── seg_info
│   ├── train_label_weight.npy
│   ├── train_resampled_scene_idxs.npy
│   ├── val_label_weight.npy
│   ├── val_resampled_scene_idxs.npy
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
├── scannet_infos_test.pkl
```

- `seg_info`：为支持语义分割任务所生成的信息文件。
    - `train_label_weight.npy`：每一语义类别的权重系数。因为 ScanNet 中属于不同类的点的数量相差很大，一个常见的操作是在计算损失时对不同类别进行加权 (label re-weighting) 以得到更好的分割性能。
    - `train_resampled_scene_idxs.npy`：每一个场景 (房间) 的重采样标签。在训练过程中，我们依据每个场景的点的数量，会对其进行不同次数的重采样，以保证训练数据均衡。

## 训练流程

ScanNet 上 3D 语义分割的一种典型数据载入流程如下所示：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39),
        max_cat_id=40),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.5,
        ignore_index=len(class_names),
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]
```

- `PointSegClassMapping`：在训练过程中，只有被使用的类别的序号会被映射到类似 [0, 20) 范围内的类别标签。其余的类别序号会被转换为 `ignore_index` 所制定的忽略标签，在本例中是 `20`。
- `IndoorPatchPointSample`：从输入点云中裁剪一个含有固定数量点的小块 (patch)。`block_size` 指定了裁剪块的边长，在 ScanNet 上这个数值一般设置为 `1.5`。
- `NormalizePointsColor`：将输入点的颜色信息归一化，通过将 RGB 值除以 `255` 来实现。

## 度量指标

通常我们使用平均交并比 (mean Intersection over Union, mIoU) 作为 ScanNet 语义分割任务的度量指标。
具体而言，我们先计算所有类别的 IoU，然后取平均值作为 mIoU。
更多实现细节请参考 [seg_eval.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/seg_eval.py)。

## 在测试集上测试并提交结果

默认情况下，MMDet3D 的代码是在训练集上进行模型训练，然后在验证集上进行模型测试。
如果你也想在在线基准上测试模型的性能，请在测试命令中加上 `--format-only` 的标记，同时也要将 ScanNet 数据集[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/scannet_seg-3d-20class.py#L126)中的 `ann_file=data_root + 'scannet_infos_val.pkl'` 改成 `ann_file=data_root + 'scannet_infos_test.pkl'`。
请记得通过 `txt_prefix` 来指定想要保存测试结果的文件夹名称。

以 PointNet++ (SSG) 在 ScanNet 上的测试为例，你可以运行以下命令来完成测试结果的保存：

```
./tools/dist_test.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py \
    work_dirs/pointnet2_ssg/latest.pth --format-only \
    --eval-options txt_prefix=work_dirs/pointnet2_ssg/test_submission
```

在保存测试结果后，你可以将该文件夹压缩，然后提交到 [ScanNet 在线测试服务器](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d)上进行验证。
