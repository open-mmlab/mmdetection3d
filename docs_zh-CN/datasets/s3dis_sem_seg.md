# 3D 语义分割 S3DIS 数据集

## 数据集的准备

对于数据集准备的整体流程，请参考 S3DIS 的[指南](https://github.com/open-mmlab/mmdetection3d/blob/master/data/s3dis/README.md/)。

### 提取 S3DIS 数据

通过从原始数据中提取 S3DIS 数据，我们将点云数据读取并保存下相关的标注信息，例如语义分割标签和实例分割标签。

数据提取前的目录结构应该如下所示：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── s3dis
│   │   ├── meta_data
│   │   ├── Stanford3dDataset_v1.2_Aligned_Version
│   │   │   ├── Area_1
│   │   │   │   ├── conferenceRoom_1
│   │   │   │   ├── office_1
│   │   │   │   ├── ...
│   │   │   ├── Area_2
│   │   │   ├── Area_3
│   │   │   ├── Area_4
│   │   │   ├── Area_5
│   │   │   ├── Area_6
│   │   ├── indoor3d_util.py
│   │   ├── collect_indoor3d_data.py
│   │   ├── README.md
```

在 `Stanford3dDataset_v1.2_Aligned_Version` 目录下，所有房间依据所属区域被分为 6 组。
我们通常使用 5 个区域进行训练，然后在余下 1 个区域上进行测试 (被余下的 1 个区域通常为区域 5)。
在每个区域的目录下包含有多个房间的文件夹，每个文件夹是一个房间的原始点云数据和相关的标注信息。
例如，在 `Area_1/office_1` 目录下的文件如下所示：

- `office_1.txt`：一个 txt 文件存储着原始点云数据每个点的坐标和颜色信息。
- `Annotations/`：这个文件夹里包含有此房间中实例物体的信息 (以 txt 文件的形式存储)。每个 txt 文件表示一个实例，例如：
    - `chair_1.txt`：存储有该房间中一把椅子的点云数据。

    如果我们将 `Annotations/` 下的所有 txt 文件合并起来，得到的点云就和 `office_1.txt` 中的点云是一致的。

你可以通过 `python collect_indoor3d_data.py` 指令进行 S3DIS 数据的提取。
主要步骤包括：

- 从原始 txt 文件中读取点云数据、语义分割标签和实例分割标签。
- 将点云数据和相关标注文件存储下来。

这其中的核心函数 `indoor3d_util.py` 中的 `export` 函数实现如下：

```python
def export(anno_path, out_filename):
    """将原始数据集的文件转化为点云、语义分割标签和实例分割掩码文件。
    我们将同一房间中所有实例的点进行聚合。

    参数列表:
        anno_path (str): 标注信息的路径，例如 Area_1/office_2/Annotations/
        out_filename (str): 保存点云和标签的路径
        file_format (str): txt 或 numpy，指定保存的文件格式

    注意:
        点云在处理过程中被整体移动了，保存下的点最小位于原点 (即没有负数坐标值)
    """
    points_list = []
    ins_idx = 1  # 实例标签从 1 开始，因此最终实例标签为 0 的点就是无标注的点

    # `anno_path` 的一个例子：Area_1/office_1/Annotations
    # 其中以 txt 文件存储有该房间中所有实例物体的点云
    for f in glob.glob(osp.join(anno_path, '*.txt')):
        # get class name of this instance
        one_class = osp.basename(f).split('_')[0]
        if one_class not in class_names:  # 某些房间有 'staris' 类物体
            one_class = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * class2label[one_class]
        ins_labels = np.ones((points.shape[0], 1)) * ins_idx
        ins_idx += 1
        points_list.append(np.concatenate([points, labels, ins_labels], 1))

    data_label = np.concatenate(points_list, 0)  # [N, 8], (pts, rgb, sem, ins)
    # 将点云对齐到原点
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int))
    np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int))

```

上述代码中，我们读取 `Annotations/` 下的所有点云实例，将其合并得到整体房屋的点云，同时生成语义/实例分割的标签。
在提取完每个房间的数据后，点云、语义分割和实例分割的标签文件应以 `.npy` 的格式被保存下来。

### 创建数据集

```shell
python tools/create_data.py s3dis --root-path ./data/s3dis \
--out-dir ./data/s3dis --extra-tag s3dis
```

上述指令首先读取以 `.npy` 格式存储的点云、语义分割和实例分割标签文件，然后进一步将它们以 `.bin` 格式保存。
同时，每个区域 `.pkl` 格式的信息文件也会被保存下来。

数据预处理后的目录结构如下所示：

```
s3dis
├── meta_data
├── indoor3d_util.py
├── collect_indoor3d_data.py
├── README.md
├── Stanford3dDataset_v1.2_Aligned_Version
├── s3dis_data
├── points
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── seg_info
│   ├── Area_1_label_weight.npy
│   ├── Area_1_resampled_scene_idxs.npy
│   ├── Area_2_label_weight.npy
│   ├── Area_2_resampled_scene_idxs.npy
│   ├── Area_3_label_weight.npy
│   ├── Area_3_resampled_scene_idxs.npy
│   ├── Area_4_label_weight.npy
│   ├── Area_4_resampled_scene_idxs.npy
│   ├── Area_5_label_weight.npy
│   ├── Area_5_resampled_scene_idxs.npy
│   ├── Area_6_label_weight.npy
│   ├── Area_6_resampled_scene_idxs.npy
├── s3dis_infos_Area_1.pkl
├── s3dis_infos_Area_2.pkl
├── s3dis_infos_Area_3.pkl
├── s3dis_infos_Area_4.pkl
├── s3dis_infos_Area_5.pkl
├── s3dis_infos_Area_6.pkl
```

- `points/xxxxx.bin`：提取的点云数据。
- `instance_mask/xxxxx.bin`：每个点云的实例标签，取值范围为 [0, ${实例个数}]，其中 0 代表未标注的点。
- `semantic_mask/xxxxx.bin`：每个点云的语义标签，取值范围为 [0, 12]。
- `s3dis_infos_Area_1.pkl`：区域 1 的数据信息，每个房间的详细信息如下：
    - info['point_cloud']: {'num_features': 6, 'lidar_idx': sample_idx}.
    - info['pts_path']: `points/xxxxx.bin` 点云的路径。
    - info['pts_instance_mask_path']: `instance_mask/xxxxx.bin` 实例标签的路径。
    - info['pts_semantic_mask_path']: `semantic_mask/xxxxx.bin` 语义标签的路径。
- `seg_info`：为支持语义分割任务所生成的信息文件。
    - `Area_1_label_weight.npy`：每一语义类别的权重系数。因为 S3DIS 中属于不同类的点的数量相差很大，一个常见的操作是在计算损失时对不同类别进行加权 (label re-weighting) 以得到更好的分割性能。
    - `Area_1_resampled_scene_idxs.npy`：每一个场景 (房间) 的重采样标签。在训练过程中，我们依据每个场景的点的数量，会对其进行不同次数的重采样，以保证训练数据均衡。

## 训练流程

S3DIS 上 3D 语义分割的一种典型数据载入流程如下所示：

```python
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
num_points = 4096
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
        valid_cat_ids=tuple(range(len(class_names))),
        max_cat_id=13),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.0,
        ignore_index=None,
        use_normalized_coord=True,
        enlarge_size=None,
        min_unique_num=num_points // 4,
        eps=0.0),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.141592653589793, 3.141592653589793],  # [-pi, pi]
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomJitterPoints',
        jitter_std=[0.01, 0.01, 0.01],
        clip_range=[-0.05, 0.05]),
    dict(type='RandomDropPointsColor', drop_ratio=0.2),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]
```

- `PointSegClassMapping`：在训练过程中，只有被使用的类别的序号会被映射到类似 [0, 13) 范围内的类别标签。其余的类别序号会被转换为 `ignore_index` 所制定的忽略标签，在本例中是 `13`。
- `IndoorPatchPointSample`：从输入点云中裁剪一个含有固定数量点的小块 (patch)。`block_size` 指定了裁剪块的边长，在 S3DIS 上这个数值一般设置为 `1.0`。
- `NormalizePointsColor`：将输入点的颜色信息归一化，通过将 RGB 值除以 `255` 来实现。
- 数据增广：
    - `GlobalRotScaleTrans`：对输入点云进行随机旋转和放缩变换。
    - `RandomJitterPoints`：通过对每一个点施加不同的噪声向量以实现对点云的随机扰动。
    - `RandomDropPointsColor`：以 `drop_ratio` 的概率随机将点云的颜色值全部置零。

## 度量指标

通常我们使用平均交并比 (mean Intersection over Union, mIoU) 作为 ScanNet 语义分割任务的度量指标。
具体而言，我们先计算所有类别的 IoU，然后取平均值作为 mIoU。
更多实现细节请参考 [seg_eval.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/seg_eval.py)。

正如在 `提取 S3DIS 数据` 一节中所提及的，S3DIS 通常在 5 个区域上进行训练，然后在余下的 1 个区域上进行测试。但是在其他论文中，也有不同的划分方式。
为了便于灵活划分训练和测试的子集，我们首先定义子数据集 (sub-dataset) 来表示每一个区域，然后根据区域划分对其进行合并，以得到完整的训练集。
以下是在区域 1、2、3、4、6 上训练并在区域 5 上测试的一个配置文件例子：

```python
dataset_type = 'S3DISSegDataset'
data_root = './data/s3dis/'
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
train_area = [1, 2, 3, 4, 6]
test_area = 5
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[
            data_root + f's3dis_infos_Area_{i}.pkl' for i in train_area
        ],
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        ignore_index=len(class_names),
        scene_idxs=[
            data_root + f'seg_info/Area_{i}_resampled_scene_idxs.npy'
            for i in train_area
        ]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=data_root + f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names),
        scene_idxs=data_root +
        f'seg_info/Area_{test_area}_resampled_scene_idxs.npy'))
```

可以看到，我们通过将多个相应路径构成的列表 (list) 输入 `ann_files` 和 `scene_idxs` 以实现训练测试集的划分。
如果修改训练测试区域的划分，只需要简单修改 `train_area` 和 `test_area` 即可。
