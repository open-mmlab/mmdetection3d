# ScanNet

## Dataset preparation

For the overall process, please refer to the [README](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/) page for ScanNet.

### Export ScanNet data

By exporting ScanNet data, we load the raw point cloud data and generate the relevant annotations including semantic label, instance label and ground truth bounding boxes.

```shell
python batch_load_scannet_data.py
```

The directory structure before data preparation should be as below

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

Under folder `scans` there are overall 1201 train and 312 validation folders in which raw point cloud data and relevant annotations are saved. For instance, under folder `scene0001_01` the files are as below:

- `scene0001_01_vh_clean_2.ply`: Mesh file including raw point cloud data.
- `scene0001_01.aggregation.json`: Aggregation file including object id, segments id and label.
- `scene0001_01_vh_clean_2.0.010000.segs.json`: Segmentation file including segments id and vertex.
- `scene0001_01.txt`: Meta file including axis-aligned matrix, etc.
- `scene0001_01_vh_clean_2.labels.ply`

Export ScanNet data by running `python batch_load_scannet_data.py`. The main steps include:

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
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        agg_file (str): Path of the agg_file.
        seg_file (str): Path of the seg_file.
        meta_file (str): Path of the meta_file.
        label_map_file (str): Path of the label_map_file.
        output_file (str): Path of the output folder.
            Default: None.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which containts the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """

    # label map file: ./data/scannet/meta_data/scannetv2-labels.combined.tsv
    # the various label standards in the label map file, e.g. 'nyu40id'
    label_map = scannet_utils.read_label_mapping(
        label_map_file, label_from='raw_category', label_to='nyu40id')
    # load raw point cloud data, 6-dims feature: XYZRGB
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
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
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]],
                                           axis=1)

    # Load semantic and instance labels
    if not test_mode:
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
        instance_ids = np.zeros(
            shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                # object_id is 1-indexed, i.e. 1,2,3,.,,,.NUM_INSTANCES
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]
        # bbox format is [x, y, z, dx, dy, dz, label_id]
        # [x, y, z] is gravity center of bbox, [dx, dy, dz] is axis-aligned
        # [label_id] is semantic label id in 'nyu40id' standard
        # Note: since 3d bbox is axis-aligned, the yaw is 0.
        unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs,
                                        object_id_to_label_id, instance_ids)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs,
                                      object_id_to_label_id, instance_ids)
    ...

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, object_id_to_label_id, axis_align_matrix

```

After exporting each scan, the raw point cloud could be downsampled, e.g. to 50000. In addition, invalid semantic label outside of `nyu40id` standard should be filtered. Finally, the point cloud data, semantic labels, instance labels and ground truth bounding boxes should be saved in `.npy` files. The overall `export_one_scan` procedure is as follows:

```python
def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False):
    ...

    # Export original files
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode)

    # filter invalid classes
    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

    # raw point cloud downsample
    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

    # save point cloud and annotations
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    if not test_mode:
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy',
                unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy',
                axis_align_matrix)
```

### Create dataset

```shell
python tools/create_data.py scannet --root-path ./data/scannet \
--out-dir ./data/scannet --extra-tag scannet
```

The above exported point cloud file, semantic label file and instance label file are further saved in `.bin` format. Meanwhile `.pkl` info files are also generated for train or validation. The core function `process_single_scene` of getting data infos is as follows.

```python
def process_single_scene(sample_idx):

    # save point cloud, instance label and semantic label in .bin file respectively, get info['pts_path'], info['pts_instance_mask_path'] and info['pts_semantic_mask_path']
    ...

    # get annotations
    if has_label:
        annotations = {}
        # box is of shape [k, 6 + class]
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
            # default names are given to aligned bbox for compatibility
            # we also save unaligned bbox info with marked names
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

The directory structure after process should be as below

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

- `points/xxxxx.bin`: The `axis-unaligned` point cloud data after downsample. Note: the point would be axis-aligned in pre-processing `GlobalAlignment` of 3d detection task.
- `instance_mask/xxxxx.bin`: The instance label for each point, value range: [0, NUM_INSTANCES], 0: unannotated.
- `semantic_mask/xxxxx.bin`: The semantic label for each point, value range: [1, 40], i.e. `nyu40id` standard. Note: the `nyu40id` id will be mapped to train id in train pipeline `PointSegClassMapping`.
- `scannet_infos_train.pkl`: The train data infos, the detailed info of each scan is as follows:
    - info['point_cloud']: {'num_features': 6, 'lidar_idx': sample_idx}.
    - info['pts_path']: The path of `points/xxxxx.bin`.
    - info['pts_instance_mask_path']: The path of `instance_mask/xxxxx.bin`.
    - info['pts_semantic_mask_path']: The path of `semantic_mask/xxxxx.bin`.
    - info['annos']: The annotations of each scan.
        - annotations['gt_num']: The number of ground truth.
        - annotations['name']： The semantic name of all ground truths, e.g. `chair`.
        - annotations['location']: The gravity center of axis-aligned 3d bounding box. Shape: [K, 3], K is the number of ground truth.
        - annotations['dimensions']: The dimensions of axis-aligned 3d bounding box, i.e. x_size, y_size, z_size, shape: [K, 3].
        - annotations['gt_boxes_upright_depth']: Axis-aligned 3d bounding box, each bounding box is x, y, z, x_size, y_size, z_size, shape: [K, 6].
        - annotations['unaligned_location']: The gravity center of axis-unaligned 3d bounding box.
        - annotations['unaligned_dimensions']: The dimensions of axis-unaligned 3d bounding box.
        - annotations['unaligned_gt_boxes_upright_depth']: Axis-unaligned 3d bounding box.
        - annotations['index']: The index of all ground truths, i.e. [0, K).
        - annotations['class']: The train class id of each bounding box, value range: [0, 18), shape: [K, ].


## Train pipeline

A typical train pipeline of ScanNet for 3d detection is as below.

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
- `GlobalAlignment`: The previous point cloud would be axis-aligned using the axis-aligned matrix.
- `PointSegClassMapping`: Only the valid category id will be mapped to train class label id like [0, 18).
- Data augmentation:
    - `IndoorPointSample`: downsample input point cloud.
    - `RandomFlip3D`: randomly flip input point cloud horizontally or vertically.
    - `GlobalRotScaleTrans`: rotate input point cloud, usually [-5, 5] degree.

## Metrics

Typically mean average precision (mAP) is used for evaluation on ScanNet, e.g. `mAP@0.25` and `mAP@0.5`. In detail, a generic functions to compute precision and recall for 3d object detection for multiple classes is called, please refer to [indoor_eval](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/indoor_eval.py).
As introduced in section `Export ScanNet data`, all ground truth 3d bounding box are axis-aligned, i.e. the yaw is zero. So the yaw target of network predicted 3d bounding box is also zero and axis-aligned 3d non-maximum suppression (NMS) is adopted during post-processing without reagrd to rotation.
