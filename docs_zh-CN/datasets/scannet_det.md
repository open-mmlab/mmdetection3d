# 3D 目标检测 Scannet 数据集

## 数据集准备

请参考 ScanNet 的[指南](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/) 以查看总体流程。

### 提取 ScanNet 点云数据

通过提取By exporting ScanNet data, we load the raw point cloud data and generate the relevant annotations including semantic labels, instance labels and ground truth bounding boxes.

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
- `scene0001_01.aggregation.json`: 包含物体 id，分割部分 id，和标签的标注文件。
- `scene0001_01_vh_clean_2.0.010000.segs.json`: 包含分割部分 id 和顶点的分割标注文件。
- `scene0001_01.txt`: 包括与坐标轴平行的矩阵等的元文件。
- `scene0001_01_vh_clean_2.labels.ply`：包含每个顶点类别的标注文件。

通过运行 `python batch_load_scannet_data.py` 来提取 ScanNet 数据。主要步骤包括：

- Export original files to point cloud, instance label, semantic label and bounding box file.
- Downsample raw point cloud and filter invalid classes.
- Save point cloud data and relevant annotation files.

And the core function `export` in `load_scannet_data.py` is as follows:

```python
def export(mesh_file,
           agg_file,
           seg_file,
           meta_file,
           label_map_file,
           output_file=None,
           test_mode=False):

    # label map file: ./data/scannet/meta_data/scannetv2-labels.combined.tsv
    # the various label standards in the label map file, e.g. 'nyu40id'
    label_map = scannet_utils.read_label_mapping(
        label_map_file, label_from='raw_category', label_to='nyu40id')
    # load raw point cloud data, 6-dims feature: XYZRGB
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix: a 4x4 transformation matrix
    # transform raw points in sensor coordinate system to a coordinate system
    # which is axis-aligned with the length/width of the room
    lines = open(meta_file).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # perform global alignment of mesh vertices
    pts = np.ones((mesh_vertices.shape[0], 4))
    # raw point cloud in homogeneous coordinats, each row: [x, y, z, 1]
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    # transform raw mesh vertices to aligned mesh vertices
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]],
                                           axis=1)

    # Load semantic and instance labels
    if not test_mode:
        # each object has one semantic label and consists of several segments
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        # many points may belong to the same segment
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                # each point has one semantic label
                label_ids[verts] = label_id
        instance_ids = np.zeros(
            shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                # object_id is 1-indexed, i.e. 1,2,3,.,,,.NUM_INSTANCES
                # each point belongs to one object
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]
        # bbox format is [x, y, z, dx, dy, dz, label_id]
        # [x, y, z] is gravity center of bbox, [dx, dy, dz] is axis-aligned
        # [label_id] is semantic label id in 'nyu40id' standard
        # Note: since 3D bbox is axis-aligned, the yaw is 0.
        unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs,
                                        object_id_to_label_id, instance_ids)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs,
                                      object_id_to_label_id, instance_ids)
    ...

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, object_id_to_label_id, axis_align_matrix

```

After exporting each scan, the raw point cloud could be downsampled, e.g. to 50000, if the number of points is too large (the raw point cloud won't be downsampled if it's also used in 3d semantic segmentation task). In addition, invalid semantic labels outside of `nyu40id` standard or optional `DONOT CARE` classes should be filtered. Finally, the point cloud data, semantic labels, instance labels and ground truth bounding boxes should be saved in `.npy` files.

### Export ScanNet RGB data

By exporting ScanNet RGB data, for each scene we load a set of RGB images with corresponding 4x4 pose matrices, and a single 4x4 camera intrinsic matrix. Note, that this step is optional and can be skipped if multi-view detection is not planned to use.

```shell
python extract_posed_images.py
```

Each of 1201 train, 312 validation and 100 test scenes contains a single `.sens` file. For instance, for scene `0001_01` we have `data/scannet/scans/scene0001_01/0001_01.sens`. For this scene all images and poses are extracted to `data/scannet/posed_images/scene0001_01`. Specifically, there will be 300 image files xxxxx.jpg, 300 camera pose files xxxxx.txt and a single `intrinsic.txt` file. Typically, single scene contains several thousand images. By default, we extract only 300 of them with resulting weight of <100 Gb. To extract more images, use `--max-images-per-scene` parameter.

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

- `points/xxxxx.bin`: The `axis-unaligned` point cloud data after downsample. Since ScanNet 3D detection task takes axis-aligned point clouds as input, while ScanNet 3D semantic segmentation task takes unaligned points, we choose to store unaligned points and their axis-align transform matrix. Note: the points would be axis-aligned in pre-processing pipeline `GlobalAlignment` of 3D detection task.
- `instance_mask/xxxxx.bin`: 每个点的实例标签，值的范围为：[0, NUM_INSTANCES]，其中 0 表示没有标注。
- `semantic_mask/xxxxx.bin`: 每个点的语义标签，值的范围为：[1, 40], 也就是 `nyu40id` 的标准。请注意：在训练流程 `PointSegClassMapping` 中，`nyu40id` 的 id 会被映射到训练 id。
- `posed_images/scenexxxx_xx`: `.jpg` 图像的集合，还包含 `.txt` 格式的 4x4 相机姿态和单个 `.txt` 格式的相机内参矩阵文件。
- `scannet_infos_train.pkl`: The train data infos, the detailed info of each scan is as follows:
    - info['point_cloud']: {'num_features': 6, 'lidar_idx': sample_idx}.
    - info['pts_path']: The path of `points/xxxxx.bin`.
    - info['pts_instance_mask_path']: The path of `instance_mask/xxxxx.bin`.
    - info['pts_semantic_mask_path']: The path of `semantic_mask/xxxxx.bin`.
    - info['annos']: The annotations of each scan.
        - annotations['gt_num']: The number of ground truths.
        - annotations['name']： The semantic name of all ground truths, e.g. `chair`.
        - annotations['location']: The gravity center of the axis-aligned 3D bounding boxes. Shape: [K, 3], K is the number of ground truths.
        - annotations['dimensions']: The dimensions of the axis-aligned 3D bounding boxes, i.e. (x_size, y_size, z_size), shape: [K, 3].
        - annotations['gt_boxes_upright_depth']: The axis-aligned 3D bounding boxes, each bounding box is (x, y, z, x_size, y_size, z_size), shape: [K, 6].
        - annotations['unaligned_location']: The gravity center of the axis-unaligned 3D bounding boxes.
        - annotations['unaligned_dimensions']: The dimensions of the axis-unaligned 3D bounding boxes.
        - annotations['unaligned_gt_boxes_upright_depth']: The axis-unaligned 3D bounding boxes.
        - annotations['index']: The index of all ground truths, i.e. [0, K).
        - annotations['class']: The train class id of the bounding boxes, value range: [0, 18), shape: [K, ].


## Training pipeline

ScanNet 上 3D 物体检测的经典流程如下：

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
    dict(type='IndoorPointSample', num_points=40000),
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

- `GlobalAlignment`：输入的点云在施加了坐标轴平行的矩阵后应被转换为坐标轴平行的。
- `PointSegClassMapping`：Only the valid category ids will be mapped to class label ids like [0, 18) during training.
- 数据增强:
    - `IndoorPointSample`：下采样输入点云。
    - `RandomFlip3D`：随机左右或前后翻转点云。
    - `GlobalRotScaleTrans`: 旋转输入点云，对于 ScanNet 角度通常落入 [-5, 5] （度）的范围；并放缩输入点云，对于 ScanNet 比例通常为 1.0；最后平移输入点云，对于 ScanNet 通常位移量为 0。

## 评估指标

Typically mean Average Precision (mAP) is used for evaluation on ScanNet, e.g. `mAP@0.25` and `mAP@0.5`. In detail, a generic function to compute precision and recall for 3D object detection for multiple classes is called, please refer to [indoor_eval](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3D/core/evaluation/indoor_eval.py).

As introduced in section `Export ScanNet data`, all ground truth 3D bounding box are axis-aligned, i.e. the yaw is zero. So the yaw target of network predicted 3D bounding box is also zero and axis-aligned 3D non-maximum suppression (NMS) is adopted during post-processing without reagrd to rotation.
