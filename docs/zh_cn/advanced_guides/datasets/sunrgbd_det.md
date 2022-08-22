# 3D 目标检测 SUN RGB-D 数据集

## 数据集的准备

对于数据集准备的整体流程，请参考 SUN RGB-D 的[指南](https://github.com/open-mmlab/mmdetection3d/blob/master/data/sunrgbd/README.md/)。

### 下载 SUN RGB-D 数据与工具包

在[这里](http://rgbd.cs.princeton.edu/data/)下载 SUN RGB-D 的数据。接下来，将 `SUNRGBD.zip`、`SUNRGBDMeta2DBB_v2.mat`、`SUNRGBDMeta3DBB_v2.mat` 和 `SUNRGBDtoolbox.zip` 移动到 `OFFICIAL_SUNRGBD` 文件夹，并解压文件。

下载完成后，数据处理之前的文件目录结构如下：

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

### 从原始数据中提取 3D 检测所需数据与标注

通过运行如下指令从原始文件中提取出 SUN RGB-D 的标注（这需要您的机器中安装了 MATLAB）：

```bash
matlab -nosplash -nodesktop -r 'extract_split;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1;quit;'
```

主要的步骤包括：

- 提取出训练集和验证集的索引文件；
- 从原始数据中提取出 3D 检测所需要的数据；
- 从原始的标注数据中提取并组织检测任务使用的标注数据。

用于从深度图中提取点云数据的 `extract_rgbd_data_v2.m` 的主要部分如下：

```matlab
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('../OFFICIAL_SUNRGBD', data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('../OFFICIAL_SUNRGBD', data.rgbpath);

% 从深度图获取点云
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
rgb(isnan(points3d(:,1)),:) = [];
points3d(isnan(points3d(:,1)),:) = [];
points3d_rgb = [points3d, rgb];

% MAT 文件比 TXT 文件小三倍。在 Python 中我们可以使用
% scipy.io.loadmat('xxx.mat')['points3d_rgb'] 来加载数据
mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
% 保存点云数据
parsave(strcat(depth_folder, mat_filename), points3d_rgb);
```

用于提取并组织检测任务标注的 `extract_rgbd_data_v1.m` 的主要部分如下：

```matlab
% 输出 2D 和 3D 包围框
data2d = data;
fid = fopen(strcat(det_label_folder, txt_filename), 'w');
for j = 1:length(data.groundtruth3DBB)
    centroid = data.groundtruth3DBB(j).centroid;  % 3D 包围框中心
    classname = data.groundtruth3DBB(j).classname;  % 类名
    orientation = data.groundtruth3DBB(j).orientation;  % 3D 包围框方向
    coeffs = abs(data.groundtruth3DBB(j).coeffs);  % 3D 包围框大小
    box2d = data2d.groundtruth2DBB(j).gtBb2D;  % 2D 包围框
    fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
end
fclose(fid);
```

上面的两个脚本调用了 SUN RGB-D 提供的[工具包](https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip)中的一些函数，如 `read3dPoints`。

使用上述脚本提取数据后，文件目录结构应如下：

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

在如下每个文件夹下，都有总计 5285 个训练集样本和 5050 个验证集样本：

- `calib`：`.txt` 后缀的相机标定文件。
- `depth`：`.mat` 后缀的点云文件，包含 xyz 坐标和 rgb 色彩值。
- `image`：`.jpg` 后缀的二维图像文件。
- `label`：`.txt` 后缀的用于检测任务的标注数据（版本二）。
- `label_v1`：`.txt` 后缀的用于检测任务的标注数据（版本一）。
- `seg_label`：`.txt` 后缀的用于分割任务的标注数据。

目前，我们使用版本一的数据用于训练与测试，因此版本二的标注并未使用。

### 创建数据集

请运行如下指令创建数据集：

```shell
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd \
--out-dir ./data/sunrgbd --extra-tag sunrgbd
```

或者，如果使用 slurm，可以使用如下指令替代：

```
bash tools/create_data.sh <job_name> sunrgbd
```

之前提到的点云数据就会被处理并以 `.bin` 格式重新存储。与此同时，`.pkl` 文件也被生成，用于存储数据标注和元信息。这一步处理中，用于生成 `.pkl` 文件的核心函数 `process_single_scene` 如下：

```python
def process_single_scene(sample_idx):
    print(f'{self.split} sample_idx: {sample_idx}')
    # 将深度图转换为点云并降采样点云
    pc_upright_depth = self.get_depth(sample_idx)
    pc_upright_depth_subsampled = random_sampling(
        pc_upright_depth, self.num_points)

    info = dict()
    pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
    info['point_cloud'] = pc_info

    # 将点云保存为 `.bin` 格式
    mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
    pc_upright_depth_subsampled.tofile(
        osp.join(self.root_dir, 'points', f'{sample_idx:06d}.bin'))

    # 存储点云存储路径
    info['pts_path'] = osp.join('points', f'{sample_idx:06d}.bin')

    # 存储图像存储路径以及其元信息
    img_path = osp.join('image', f'{sample_idx:06d}.jpg')
    image_info = {
        'image_idx': sample_idx,
        'image_shape': self.get_image_shape(sample_idx),
        'image_path': img_path
    }
    info['image'] = image_info

    # 保存标定信息
    K, Rt = self.get_calibration(sample_idx)
    calib_info = {'K': K, 'Rt': Rt}
    info['calib'] = calib_info

    # 保存所有数据标注
    if has_label:
        obj_list = self.get_label_objects(sample_idx)
        annotations = {}
        annotations['gt_num'] = len([
            obj.classname for obj in obj_list
            if obj.classname in self.cat2label.keys()
        ])
        if annotations['gt_num'] != 0:
            # 类别名称
            annotations['name'] = np.array([
                obj.classname for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            # 二维图像包围框
            annotations['bbox'] = np.concatenate([
                obj.box2d.reshape(1, 4) for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ], axis=0)
            # depth 坐标系下的三维包围框中心坐标
            annotations['location'] = np.concatenate([
                obj.centroid.reshape(1, 3) for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ], axis=0)
            # depth 坐标系下的三维包围框大小
            annotations['dimensions'] = 2 * np.array([
                [obj.l, obj.h, obj.w] for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            # depth 坐标系下的三维包围框旋转角
            annotations['rotation_y'] = np.array([
                obj.heading_angle for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            annotations['index'] = np.arange(
                len(obj_list), dtype=np.int32)
            # 类别标签（数字）
            annotations['class'] = np.array([
                self.cat2label[obj.classname] for obj in obj_list
                if obj.classname in self.cat2label.keys()
            ])
            # depth 坐标系下的三维包围框
            annotations['gt_boxes_upright_depth'] = np.stack(
                [
                    obj.box3d for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ], axis=0)  # (K,8)
        info['annos'] = annotations
    return info
```

如上数据处理后，文件目录结构应如下：

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

- `points/0xxxxx.bin`：降采样后的点云数据。
- `sunrgbd_infos_train.pkl`：训练集数据信息（标注与元信息），每个场景所含数据信息具体如下：
  - info\['point_cloud'\]：`{'num_features': 6, 'lidar_idx': sample_idx}`，其中 `sample_idx` 为该场景的索引。
  - info\['pts_path'\]：`points/0xxxxx.bin` 的路径。
  - info\['image'\]：图像路径与元信息：
    - image\['image_idx'\]：图像索引。
    - image\['image_shape'\]：图像张量的形状（即其尺寸）。
    - image\['image_path'\]：图像路径。
  - info\['annos'\]：每个场景的标注：
    - annotations\['gt_num'\]：真实物体 (ground truth) 的数量。
    - annotations\['name'\]：所有真实物体的语义类别名称，比如 `chair`（椅子）。
    - annotations\['location'\]：depth 坐标系下三维包围框的重力中心 (gravity center)，形状为 \[K, 3\]，其中 K 是真实物体的数量。
    - annotations\['dimensions'\]：depth 坐标系下三维包围框的大小，形状为 \[K, 3\]。
    - annotations\['rotation_y'\]：depth 坐标系下三维包围框的旋转角，形状为 \[K, \]。
    - annotations\['gt_boxes_upright_depth'\]：depth 坐标系下三维包围框 `(x, y, z, x_size, y_size, z_size, yaw)`，形状为 \[K, 7\]。
    - annotations\['bbox'\]：二维包围框 `(x, y, x_size, y_size)`，形状为 \[K, 4\]。
    - annotations\['index'\]：所有真实物体的索引，范围为 \[0, K)。
    - annotations\['class'\]：所有真实物体类别的标号，范围为 \[0, 10)，形状为 \[K, \]。
- `sunrgbd_infos_val.pkl`：验证集上的数据信息，与 `sunrgbd_infos_train.pkl` 格式完全一致。

## 训练流程

SUN RGB-D 上纯点云 3D 物体检测的典型流程如下：

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

点云上的数据增强

- `RandomFlip3D`：随机左右或前后翻转输入点云。
- `GlobalRotScaleTrans`：旋转输入点云，对于 SUN RGB-D 角度通常落入 \[-30, 30\] （度）的范围；并放缩输入点云，对于 SUN RGB-D 比例通常落入 \[0.85, 1.15\] 的范围；最后平移输入点云，对于 SUN RGB-D 通常位移量为 0（即不做位移）。
- `PointSample`：降采样输入点云。

SUN RGB-D 上多模态（点云和图像）3D 物体检测的典型流程如下：

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

图像上的数据增强/归一化

- `Resize`: 改变输入图像的大小, `keep_ratio=True` 意味着图像的比例不改变。
- `Normalize`: 归一化图像的 RGB 通道。
- `RandomFlip`: 随机地翻折图像。
- `Pad`: 扩大图像，默认情况下用零填充图像的边缘。

图像增强和归一化函数的实现取自 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/mmdet/datasets/pipelines)。

## 度量指标

与 ScanNet 一样，通常 mAP（全类平均精度）被用于 SUN RGB-D 的检测任务的评估，比如 `mAP@0.25` 和 `mAP@0.5`。具体来说，评估时一个通用的计算 3D 物体检测多个类别的精度和召回率的函数被调用，可以参考 [`indoor_eval.py`](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/indoor_eval.py)。

因为 SUN RGB-D 包含有图像数据，所以图像上的物体检测也是可行的。举个例子，在 ImVoteNet 中，我们首先训练了一个图像检测器，并且也使用 mAP 指标，如 `mAP@0.5`，来评估其表现。我们使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 库中的 `eval_map` 函数来计算 mAP。
