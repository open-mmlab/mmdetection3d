# 3D 目标检测 Scannet 数据集

## 数据集准备

请参考 ScanNet 的[指南](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/)以查看总体流程。

### 提取 ScanNet 点云数据

通过提取 ScanNet 数据，我们加载原始点云文件，并生成包括语义标签、实例标签和真实物体包围框在内的相关标注。

```shell
python batch_load_scannet_data.py
```

数据处理之前的文件目录结构如下：

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
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
```

在 `scans` 文件夹下总共有 1201 个训练样本文件夹和 312 个验证样本文件夹，其中存有未处理的点云数据和相关的标注。比如说，在文件夹 `scene0001_01` 下文件是这样组织的：

- `scene0001_01_vh_clean_2.ply`: 存有每个顶点坐标和颜色的网格文件。网格的顶点被直接用作未处理的点云数据。
- `scene0001_01.aggregation.json`: 包含物体 ID、分割部分 ID、标签的标注文件。
- `scene0001_01_vh_clean_2.0.010000.segs.json`: 包含分割部分 ID 和顶点的分割标注文件。
- `scene0001_01.txt`: 包括对齐矩阵等的元文件。
- `scene0001_01_vh_clean_2.labels.ply`：包含每个顶点类别的标注文件。

通过运行 `python batch_load_scannet_data.py` 来提取 ScanNet 数据。主要步骤包括：

- 从原始文件中提取出点云、实例标签、语义标签和包围框标签文件。
- 下采样原始点云并过滤掉不合法的类别。
- 保存处理后的点云数据和相关的标注文件。

`load_scannet_data.py` 中的核心函数 `export` 如下：

```python
def export(mesh_file,
           agg_file,
           seg_file,
           meta_file,
           label_map_file,
           output_file=None,
           test_mode=False):

    # 标签映射文件：./data/scannet/meta_data/scannetv2-labels.combined.tsv
    # 该标签映射文件中有多种标签标准，比如 'nyu40id'
    label_map = scannet_utils.read_label_mapping(
        label_map_file, label_from='raw_category', label_to='nyu40id')
    # 加载原始点云数据，特征包括6维：XYZRGB
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # 加载场景坐标轴对齐矩阵：一个 4x4 的变换矩阵
    # 将传感器坐标系下的原始点转化到另一个坐标系下
    # 该坐标系与房屋的两边平行（也就是与坐标轴平行）
    lines = open(meta_file).readlines()
    # 测试集的数据没有对齐矩阵
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # 对网格顶点进行全局的对齐
    pts = np.ones((mesh_vertices.shape[0], 4))
    # 同种类坐标下的原始点云，每一行的数据是 [x, y, z, 1]
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    # 将原始网格顶点转换为对齐后的顶点
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]],
                                           axis=1)

    # 加载语义与实例标签
    if not test_mode:
        # 每个物体都有一个语义标签，并且包含几个分割部分
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        # 很多点属于同一分割部分
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                # 每个点都有一个语义标签
                label_ids[verts] = label_id
        instance_ids = np.zeros(
            shape=(num_verts), dtype=np.uint32)  # 0：未标注的
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                # object_id 从 1 开始计数，比如 1,2,3,.,,,.NUM_INSTANCES
                # 每个点都属于一个物体
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]
        # 包围框格式为 [x, y, z, dx, dy, dz, label_id]
        # [x, y, z] 是包围框的重力中心, [dx, dy, dz] 是与坐标轴平行的
        # [label_id] 是 'nyu40id' 标准下的语义标签
        # 注意：因为三维包围框是与坐标轴平行的，所以旋转角是 0
        unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs,
                                        object_id_to_label_id, instance_ids)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs,
                                      object_id_to_label_id, instance_ids)
    ...

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, object_id_to_label_id, axis_align_matrix

```

在从每个场景的扫描文件提取数据后，如果原始点云点数过多，可以将其下采样（比如到 50000 个点），但在三维语义分割任务中，点云不会被下采样。此外，在 `nyu40id` 标准之外的不合法语义标签或者可选的 `DONOT CARE` 类别标签应被过滤。最终，点云文件、语义标签、实例标签和真实物体的集合应被存储于 `.npy` 文件中。

### 提取 ScanNet RGB 色彩数据（可选的）

通过提取 ScanNet RGB 色彩数据，对于每个场景我们加载 RGB 图像与配套 4x4 位姿矩阵、单个 4x4 相机内参矩阵的集合。请注意，这一步是可选的，除非要运行多视图物体检测，否则可以略去这步。

```shell
python extract_posed_images.py
```

1201 个训练样本，312 个验证样本和 100 个测试样本中的每一个都包含一个单独的 `.sens` 文件。比如说，对于场景 `0001_01` 我们有 `data/scannet/scans/scene0001_01/0001_01.sens`。对于这个场景所有图像和位姿数据都被提取至 `data/scannet/posed_images/scene0001_01`。具体来说，该文件夹下会有 300 个 xxxxx.jpg 格式的图像数据，300 个 xxxxx.txt 格式的相机位姿数据和一个单独的 `intrinsic.txt` 内参文件。通常来说，一个场景包含数千张图像。默认情况下，我们只会提取其中的 300 张，从而只占用少于 100 Gb 的空间。要想提取更多图像，请使用 `--max-images-per-scene` 参数。

### 创建数据集

```shell
python tools/create_data.py scannet --root-path ./data/scannet \
--out-dir ./data/scannet --extra-tag scannet
```

上述提取的点云文件，语义类别标注文件，和物体实例标注文件被进一步以 `.bin` 格式保存。与此同时 `.pkl` 格式的文件被生成并用于训练和验证。获取数据信息的核心函数 `process_single_scene` 如下：

```python
def process_single_scene(sample_idx):

    # 分别以 .bin 格式保存点云文件，语义类别标注文件和物体实例标注文件
    # 获取 info['pts_path']，info['pts_instance_mask_path'] 和 info['pts_semantic_mask_path']
    ...

    # 获取标注
    if has_label:
        annotations = {}
        # 包围框的形状为 [k, 6 + class]
        aligned_box_label = self.get_aligned_box_label(sample_idx)
        unaligned_box_label = self.get_unaligned_box_label(sample_idx)
        annotations['gt_num'] = aligned_box_label.shape[0]
        if annotations['gt_num'] != 0:
            aligned_box = aligned_box_label[:, :-1]  # k, 6
            unaligned_box = unaligned_box_label[:, :-1]
            classes = aligned_box_label[:, -1]  # k
            annotations['name'] = np.array([
                self.label2cat[self.cat_ids2class[classes[i]]]
                for i in range(annotations['gt_num'])
            ])
            # 为了向后兼容，默认的参数名赋予了与坐标轴平行的包围框
            # 我们同时保存了对应的与坐标轴不平行的包围框的信息
            annotations['location'] = aligned_box[:, :3]
            annotations['dimensions'] = aligned_box[:, 3:6]
            annotations['gt_boxes_upright_depth'] = aligned_box
            annotations['unaligned_location'] = unaligned_box[:, :3]
            annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
            annotations[
                'unaligned_gt_boxes_upright_depth'] = unaligned_box
            annotations['index'] = np.arange(
                annotations['gt_num'], dtype=np.int32)
            annotations['class'] = np.array([
                self.cat_ids2class[classes[i]]
                for i in range(annotations['gt_num'])
            ])
        axis_align_matrix = self.get_axis_align_matrix(sample_idx)
        annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
        info['annos'] = annotations
    return info
```

如上数据处理后，文件目录结构应如下：

```
scannet
├── meta_data
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
├── posed_images
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.txt
│   │   ├── xxxxxx.jpg
│   │   ├── intrinsic.txt
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
├── scannet_infos_test.pkl
```

- `points/xxxxx.bin`：下采样后，未与坐标轴平行（即没有对齐）的点云。因为 ScanNet 3D 检测任务将与坐标轴平行的点云作为输入，而 ScanNet 3D 语义分割任务将对齐前的点云作为输入，我们选择存储对齐前的点云和它们的对齐矩阵。请注意：在 3D 检测的预处理流程 [`GlobalAlignment`](https://github.com/open-mmlab/mmdetection3d/blob/9f0b01caf6aefed861ef4c3eb197c09362d26b32/mmdet3d/datasets/pipelines/transforms_3d.py#L423) 后，点云就都是与坐标轴平行的了。
- `instance_mask/xxxxx.bin`：每个点的实例标签，值的范围为：\[0, NUM_INSTANCES\]，其中 0 表示没有标注。
- `semantic_mask/xxxxx.bin`：每个点的语义标签，值的范围为：\[1, 40\], 也就是 `nyu40id` 的标准。请注意：在训练流程 `PointSegClassMapping` 中，`nyu40id` 的 ID 会被映射到训练 ID。
- `posed_images/scenexxxx_xx`：`.jpg` 图像的集合，还包含 `.txt` 格式的 4x4 相机姿态和单个 `.txt` 格式的相机内参矩阵文件。
- `scannet_infos_train.pkl`：训练集的数据信息，每个场景的具体信息如下：
  - info\['point_cloud'\]：`{'num_features': 6, 'lidar_idx': sample_idx}`，其中 `sample_idx` 为该场景的索引。
  - info\['pts_path'\]：`points/xxxxx.bin` 的路径。
  - info\['pts_instance_mask_path'\]：`instance_mask/xxxxx.bin` 的路径。
  - info\['pts_semantic_mask_path'\]：`semantic_mask/xxxxx.bin` 的路径。
  - info\['annos'\]：每个场景的标注。
    - annotations\['gt_num'\]：真实物体 (ground truth) 的数量。
    - annotations\['name'\]：所有真实物体的语义类别名称，比如 `chair`（椅子）。
    - annotations\['location'\]：depth 坐标系下与坐标轴平行的三维包围框的重力中心 (gravity center)，形状为 \[K, 3\]，其中 K 是真实物体的数量。
    - annotations\['dimensions'\]：depth 坐标系下与坐标轴平行的三维包围框的大小，形状为 \[K, 3\]。
    - annotations\['gt_boxes_upright_depth'\]：depth 坐标系下与坐标轴平行的三维包围框 `(x, y, z, x_size, y_size, z_size, yaw)`，形状为 \[K, 6\]。
    - annotations\['unaligned_location'\]：depth 坐标系下与坐标轴不平行（对齐前）的三维包围框的重力中心。
    - annotations\['unaligned_dimensions'\]：depth 坐标系下与坐标轴不平行的三维包围框的大小。
    - annotations\['unaligned_gt_boxes_upright_depth'\]：depth 坐标系下与坐标轴不平行的三维包围框。
    - annotations\['index'\]：所有真实物体的索引，范围为 \[0, K)。
    - annotations\['class'\]：所有真实物体类别的标号，范围为 \[0, 18)，形状为 \[K, \]。
- `scannet_infos_val.pkl`：验证集上的数据信息，与 `scannet_infos_train.pkl` 格式完全一致。
- `scannet_infos_test.pkl`：测试集上的数据信息，与 `scannet_infos_train.pkl` 格式几乎完全一致，除了缺少标注。

## 训练流程

ScanNet 上 3D 物体检测的典型流程如下：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39),
        max_cat_id=40),
    dict(type='PointSample', num_points=40000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
```

- `GlobalAlignment`：输入的点云在施加了坐标轴平行的矩阵后应被转换为与坐标轴平行的形式。
- `PointSegClassMapping`：训练中，只有合法的类别 ID 才会被映射到类别标签，比如 \[0, 18)。
- 数据增强:
  - `PointSample`：下采样输入点云。
  - `RandomFlip3D`：随机左右或前后翻转点云。
  - `GlobalRotScaleTrans`: 旋转输入点云，对于 ScanNet 角度通常落入 \[-5, 5\] （度）的范围；并放缩输入点云，对于 ScanNet 比例通常为 1.0（即不做缩放）；最后平移输入点云，对于 ScanNet 通常位移量为 0（即不做位移）。

## 评估指标

通常 mAP（全类平均精度）被用于 ScanNet 的检测任务的评估，比如 `mAP@0.25` 和 `mAP@0.5`。具体来说，评估时一个通用的计算 3D 物体检测多个类别的精度和召回率的函数被调用，可以参考 [indoor_eval](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3D/core/evaluation/indoor_eval.py)。

与在章节`提取 ScanNet 数据` 中介绍的那样，所有真实物体的三维包围框是与坐标轴平行的，也就是说旋转角为 0。因此，预测包围框的网络接受的包围框旋转角监督也是 0，且在后处理阶段我们使用适用于与坐标轴平行的包围框的非极大值抑制 (NMS) ，该过程不会考虑包围框的旋转。
