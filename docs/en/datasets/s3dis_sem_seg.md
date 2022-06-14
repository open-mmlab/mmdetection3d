# S3DIS for 3D Semantic Segmentation

## Dataset preparation

For the overall process, please refer to the [README](https://github.com/open-mmlab/mmdetection3d/blob/master/data/s3dis/README.md/) page for S3DIS.

### Export S3DIS data

By exporting S3DIS data, we load the raw point cloud data and generate the relevant annotations including semantic labels and instance labels.

The directory structure before exporting should be as below:

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

Under folder `Stanford3dDataset_v1.2_Aligned_Version`, the rooms are spilted into 6 areas. We use 5 areas for training and 1 for evaluation (typically `Area_5`). Under the directory of each area, there are folders in which raw point cloud data and relevant annotations are saved. For instance, under folder `Area_1/office_1` the files are as below:

- `office_1.txt`: A txt file storing coordinates and colors of each point in the raw point cloud data.

- `Annotations/`: This folder contains txt files for different object instances. Each txt file represents one instance, e.g.

  - `chair_1.txt`: A txt file storing raw point cloud data of one chair in this room.

  If we concat all the txt files under `Annotations/`, we will get the same point cloud as denoted by `office_1.txt`.

Export S3DIS data by running `python collect_indoor3d_data.py`. The main steps include:

- Export original txt files to point cloud, instance label and semantic label.
- Save point cloud data and relevant annotation files.

And the core function `export` in `indoor3d_util.py` is as follows:

```python
def export(anno_path, out_filename):
    """Convert original dataset files to points, instance mask and semantic
    mask files. We aggregated all the points from each instance in the room.

    Args:
        anno_path (str): path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename (str): path to save collected points and labels.
        file_format (str): txt or numpy, determines what file format to save.

    Note:
        the points are shifted before save, the most negative point is now
            at origin.
    """
    points_list = []
    ins_idx = 1  # instance ids should be indexed from 1, so 0 is unannotated

    # an example of `anno_path`: Area_1/office_1/Annotations
    # which contains all object instances in this room as txt files
    for f in glob.glob(osp.join(anno_path, '*.txt')):
        # get class name of this instance
        one_class = osp.basename(f).split('_')[0]
        if one_class not in class_names:  # some rooms have 'staris' class
            one_class = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * class2label[one_class]
        ins_labels = np.ones((points.shape[0], 1)) * ins_idx
        ins_idx += 1
        points_list.append(np.concatenate([points, labels, ins_labels], 1))

    data_label = np.concatenate(points_list, 0)  # [N, 8], (pts, rgb, sem, ins)
    # align point cloud to the origin
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int))
    np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int))

```

where we load and concatenate all the point cloud instances under `Annotations/` to form raw point cloud and generate semantic/instance labels. After exporting each room, the point cloud data, semantic labels and instance labels should be saved in `.npy` files.

### Create dataset

```shell
python tools/create_data.py s3dis --root-path ./data/s3dis \
--out-dir ./data/s3dis --extra-tag s3dis
```

The above exported point cloud files, semantic label files and instance label files are further saved in `.bin` format. Meanwhile `.pkl` info files are also generated for each area.

The directory structure after process should be as below:

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

- `points/xxxxx.bin`: The exported point cloud data.
- `instance_mask/xxxxx.bin`: The instance label for each point, value range: \[0, ${NUM_INSTANCES}\], 0: unannotated.
- `semantic_mask/xxxxx.bin`: The semantic label for each point, value range: \[0, 12\].
- `s3dis_infos_Area_1.pkl`: Area 1 data infos, the detailed info of each room is as follows:
  - info\['point_cloud'\]: {'num_features': 6, 'lidar_idx': sample_idx}.
  - info\['pts_path'\]: The path of `points/xxxxx.bin`.
  - info\['pts_instance_mask_path'\]: The path of `instance_mask/xxxxx.bin`.
  - info\['pts_semantic_mask_path'\]: The path of `semantic_mask/xxxxx.bin`.
- `seg_info`: The generated infos to support semantic segmentation model training.
  - `Area_1_label_weight.npy`: Weighting factor for each semantic class. Since the number of points in different classes varies greatly, it's a common practice to use label re-weighting to get a better performance.
  - `Area_1_resampled_scene_idxs.npy`: Re-sampling index for each scene. Different rooms will be sampled multiple times according to their number of points to balance training data.

## Training pipeline

A typical training pipeline of S3DIS for 3D semantic segmentation is as below.

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

- `PointSegClassMapping`: Only the valid category ids will be mapped to class label ids like \[0, 13) during training. Other class ids will be converted to `ignore_index` which equals to `13`.
- `IndoorPatchPointSample`: Crop a patch containing a fixed number of points from input point cloud. `block_size` indicates the size of the cropped block, typically `1.0` for S3DIS.
- `NormalizePointsColor`: Normalize the RGB color values of input point cloud by dividing `255`.
- Data augmentation:
  - `GlobalRotScaleTrans`: randomly rotate and scale input point cloud.
  - `RandomJitterPoints`: randomly jitter point cloud by adding different noise vector to each point.
  - `RandomDropPointsColor`: set the colors of point cloud to all zeros by a probability `drop_ratio`.

## Metrics

Typically mean intersection over union (mIoU) is used for evaluation on S3DIS. In detail, we first compute IoU for multiple classes and then average them to get mIoU, please refer to [seg_eval.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/seg_eval.py).

As introduced in section `Export S3DIS data`, S3DIS trains on 5 areas and evaluates on the remaining 1 area. But there are also other area split schemes in different papers.
To enable flexible combination of train-val splits, we use sub-dataset to represent one area, and concatenate them to form a larger training set. An example of training on area 1, 2, 3, 4, 6 and evaluating on area 5 is shown as below:

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

where we specify the areas used for training/validation by setting `ann_files` and `scene_idxs` with lists that include corresponding paths. The train-val split can be simply modified via changing the `train_area` and `test_area` variables.
