# SUN RGB-D for 3D Object Detection

## Dataset preparation

For the overall process, please refer to the [README](https://github.com/open-mmlab/mmdetection3d/blob/master/data/sunrgbd/README.md/) page for SUN RGB-D.

### Download SUN RGB-D data and toolbox

Download SUNRGBD data [HERE](http://rgbd.cs.princeton.edu/data/). Then, move `SUNRGBD.zip`, `SUNRGBDMeta2DBB_v2.mat`, `SUNRGBDMeta3DBB_v2.mat` and `SUNRGBDtoolbox.zip` to the `OFFICIAL_SUNRGBD` folder, unzip the zip files.

The directory structure before data preparation should be as below:

```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   ├── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   ├── SUNRGBDtoolbox
```

### Extract data and annotations for 3D detection from raw data

Extract SUN RGB-D annotation data from raw annotation data by running (this requires MATLAB installed on your machine):

```bash
matlab -nosplash -nodesktop -r 'extract_split;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1;quit;'
```

The main steps include:

- Extract train and val split.
- Extract data for 3D detection from raw data.
- Extract and format detection annotation from raw data.

The main component of `extract_rgbd_data_v2.m` which extracts point cloud data from depth map is as follows:

```matlab
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('../OFFICIAL_SUNRGBD', data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('../OFFICIAL_SUNRGBD', data.rgbpath);

% extract point cloud from depth map
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
rgb(isnan(points3d(:,1)),:) = [];
points3d(isnan(points3d(:,1)),:) = [];
points3d_rgb = [points3d, rgb];

% MAT files are 3x smaller than TXT files. In Python we can use
% scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
% save point cloud data
parsave(strcat(depth_folder, mat_filename), points3d_rgb);
```

The main component of `extract_rgbd_data_v1.m` which extracts annotation is as follows:

```matlab
% Write 2D and 3D box label
data2d = data;
fid = fopen(strcat(det_label_folder, txt_filename), 'w');
for j = 1:length(data.groundtruth3DBB)
    centroid = data.groundtruth3DBB(j).centroid;  % 3D bbox center
    classname = data.groundtruth3DBB(j).classname;  % class name
    orientation = data.groundtruth3DBB(j).orientation;  % 3D bbox orientation
    coeffs = abs(data.groundtruth3DBB(j).coeffs);  % 3D bbox size
    box2d = data2d.groundtruth2DBB(j).gtBb2D;  % 2D bbox
    fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
end
fclose(fid);
```

The above two scripts call functions such as `read3dPoints` from the [toolbox](https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip) provided by SUN RGB-D.

The directory structure after extraction should be as follows.

```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   ├── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   ├── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt
│   ├── val_data_idx.txt
```

Under each following folder there are overall 5285 train files and 5050 val files:

- `calib`: Camera calibration information in `.txt`
- `depth`: Point cloud saved in `.mat` (xyz+rgb)
- `image`: Image data in `.jpg`
- `label`: Detection annotation data in `.txt` (version 2)
- `label_v1`: Detection annotation data in `.txt` (version 1)
- `seg_label`: Segmentation annotation data in `.txt`

Currently, we use v1 data for training and testing, so the version 2 labels are unused.

### Create dataset

Please run the command below to create the dataset.

```shell
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd \
--out-dir ./data/sunrgbd --extra-tag sunrgbd
```

or (if in a slurm environment)

```
bash tools/create_data.sh <job_name> sunrgbd
```

The above point cloud data are further saved in `.bin` format. Meanwhile `.pkl` info files are also generated for saving annotation and metadata. The core function `process_single_scene` of getting data infos is as follows.

```python
def process_single_scene(sample_idx):
    print(f'{self.split} sample_idx: {sample_idx}')
    # convert depth to points
    pc_upright_depth = self.get_depth(sample_idx)
    pc_upright_depth_subsampled = random_sampling(
        pc_upright_depth, self.num_points)

    info = dict()
    pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
    info['point_cloud'] = pc_info

    # save point cloud data in `.bin` format
    mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
    pc_upright_depth_subsampled.tofile(
        osp.join(self.root_dir, 'points', f'{sample_idx:06d}.bin'))

    # save point cloud file path
    info['pts_path'] = osp.join('points', f'{sample_idx:06d}.bin')

    # save image file path and metainfo
    img_path = osp.join('image', f'{sample_idx:06d}.jpg')
    image_info = {
        'image_idx': sample_idx,
        'image_shape': self.get_image_shape(sample_idx),
        'image_path': img_path
    }
    info['image'] = image_info

    # save calibration information
    K, Rt = self.get_calibration(sample_idx)
    calib_info = {'K': K, 'Rt': Rt}
    info['calib'] = calib_info

    # save all annotation
    if has_label:
        obj_list = self.get_label_objects(sample_idx)
        annotations = {}
        annotations['gt_num'] = len([
            obj.classname for obj in obj_list
            if obj.classname in self.cat2label.keys()
        ])
        if annotations['gt_num'] != 0:
            # class name
            annotations['name'] = np.array([
                obj.classname for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            # 2D image bounding boxes
            annotations['bbox'] = np.concatenate([
                obj.box2d.reshape(1, 4) for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ], axis=0)
            # 3D bounding box center location (in depth coordinate system)
            annotations['location'] = np.concatenate([
                obj.centroid.reshape(1, 3) for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ], axis=0)
            # 3D bounding box dimension/size (in depth coordinate system)
            annotations['dimensions'] = 2 * np.array([
                [obj.l, obj.h, obj.w] for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            # 3D bounding box rotation angle/yaw angle (in depth coordinate system)
            annotations['rotation_y'] = np.array([
                obj.heading_angle for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            annotations['index'] = np.arange(
                len(obj_list), dtype=np.int32)
            # class label (number)
            annotations['class'] = np.array([
                self.cat2label[obj.classname] for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            # 3D bounding box (in depth coordinate system)
            annotations['gt_boxes_upright_depth'] = np.stack(
                [
                    obj.box3d for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ], axis=0)  # (K,8)
        info['annos'] = annotations
    return info
```

The directory structure after processing should be as follows.

```
sunrgbd
├── README.md
├── matlab
│   ├── ...
├── OFFICIAL_SUNRGBD
│   ├── ...
├── sunrgbd_trainval
│   ├── ...
├── points
├── sunrgbd_infos_train.pkl
├── sunrgbd_infos_val.pkl
```

- `points/0xxxxx.bin`: The point cloud data after downsample.
- `sunrgbd_infos_train.pkl`: The train data infos, the detailed info of each scene is as follows:
  - info\['point_cloud'\]: `·`{'num_features': 6, 'lidar_idx': sample_idx}`, where `sample_idx\` is the index of the scene.
  - info\['pts_path'\]: The path of `points/0xxxxx.bin`.
  - info\['image'\]: The image path and metainfo:
    - image\['image_idx'\]: The index of the image.
    - image\['image_shape'\]: The shape of the image tensor.
    - image\['image_path'\]: The path of the image.
  - info\['annos'\]: The annotations of each scene.
    - annotations\['gt_num'\]: The number of ground truths.
    - annotations\['name'\]: The semantic name of all ground truths, e.g. `chair`.
    - annotations\['location'\]: The gravity center of the 3D bounding boxes in depth coordinate system. Shape: \[K, 3\], K is the number of ground truths.
    - annotations\['dimensions'\]: The dimensions of the 3D bounding boxes in depth coordinate system, i.e. `(x_size, y_size, z_size)`, shape: \[K, 3\].
    - annotations\['rotation_y'\]: The yaw angle of the 3D bounding boxes in depth coordinate system. Shape: \[K, \].
    - annotations\['gt_boxes_upright_depth'\]: The 3D bounding boxes in depth coordinate system, each bounding box is `(x, y, z, x_size, y_size, z_size, yaw)`, shape: \[K, 7\].
    - annotations\['bbox'\]: The 2D bounding boxes, each bounding box is `(x, y, x_size, y_size)`, shape: \[K, 4\].
    - annotations\['index'\]: The index of all ground truths, range \[0, K).
    - annotations\['class'\]: The train class id of the bounding boxes, value range: \[0, 10), shape: \[K, \].
- `sunrgbd_infos_val.pkl`: The val data infos, which shares the same format as `sunrgbd_infos_train.pkl`.

## Train pipeline

A typical train pipeline of SUN RGB-D for point cloud only 3D detection is as follows.

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadAnnotations3D'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

Data augmentation for point clouds:

- `RandomFlip3D`: randomly flip the input point cloud horizontally or vertically.
- `GlobalRotScaleTrans`: rotate the input point cloud, usually in the range of \[-30, 30\] (degrees) for SUN RGB-D; then scale the input point cloud, usually in the range of \[0.85, 1.15\] for SUN RGB-D; finally translate the input point cloud, usually by 0 for SUN RGB-D (which means no translation).
- `PointSample`: downsample the input point cloud.

A typical train pipeline of SUN RGB-D for multi-modality (point cloud and image) 3D detection is as follows.

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'points', 'gt_bboxes_3d',
            'gt_labels_3d'
        ])
]
```

Data augmentation/normalization for images:

- `Resize`: resize the input image, `keep_ratio=True` means the ratio of the image is kept unchanged.
- `Normalize`: normalize the RGB channels of the input image.
- `RandomFlip`: randomly flip the input image.
- `Pad`: pad the input image with zeros by default.

The image augmentation and normalization functions are implemented in [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/mmdet/datasets/pipelines).

## Metrics

Same as ScanNet, typically mean Average Precision (mAP) is used for evaluation on SUN RGB-D, e.g. `mAP@0.25` and `mAP@0.5`. In detail, a generic function to compute precision and recall for 3D object detection for multiple classes is called, please refer to [indoor_eval](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/indoor_eval.py).

Since SUN RGB-D consists of image data, detection on image data is also feasible. For instance, in ImVoteNet, we first train an image detector, and we also use mAP for evaluation, e.g. `mAP@0.5`. We use the `eval_map` function from [MMDetection](https://github.com/open-mmlab/mmdetection) to calculate mAP.
