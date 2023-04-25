# SUN RGB-D 数据集

## 数据集的准备

对于数据集准备的整体流程，请参考 SUN RGB-D 的[指南](https://github.com/open-mmlab/mmdetection3d/blob/master/data/sunrgbd/README.md)。

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

之前提到的点云数据就会被处理并以 `.bin` 格式重新存储。与此同时，`.pkl` 文件也被生成，用于存储数据标注和元信息。

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

- `points/xxxxxx.bin`：降采样后的点云数据。
- `sunrgbd_infos_train.pkl`：训练集数据信息（标注与元信息），每个场景所含数据信息具体如下：
  - info\['lidar_points'\]：字典包含了与激光雷达点相关的信息。
    - info\['lidar_points'\]\['num_pts_feats'\]：点的特征维度。
    - info\['lidar_points'\]\['lidar_path'\]：激光雷达点云数据的文件名。
  - info\['images'\]：字典包含了与图像数据相关的信息。
    - info\['images'\]\['CAM0'\]\['img_path'\]：图像的文件名。
    - info\['images'\]\['CAM0'\]\['depth2img'\]：深度到图像的变换矩阵，形状为 (4, 4)。
    - info\['images'\]\['CAM0'\]\['height'\]：图像的高。
    - info\['images'\]\['CAM0'\]\['width'\]：图像的宽。
  - info\['instances'\]：由字典组成的列表，包含了该帧的所有标注信息。每个字典与单个实例的标注相关。对于其中的第 i 个实例，我们有：
    - info\['instances'\]\[i\]\['bbox_3d'\]：长度为 7 的列表，表示深度坐标系下的 3D 边界框。
    - info\['instances'\]\[i\]\['bbox'\]：长度为 4 的列表，以 (x1, y1, x2, y2) 的顺序表示实例的 2D 边界框。
    - info\['instances'\]\[i\]\['bbox_label_3d'\]：整数表示实例的 3D 标签，-1 表示忽略该类别。
    - info\['instances'\]\[i\]\['bbox_label'\]：整数表示实例的 2D 标签，-1 表示忽略该类别。
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

点云上的数据增强

- `RandomFlip3D`：随机左右或前后翻转输入点云。
- `GlobalRotScaleTrans`：旋转输入点云，对于 SUN RGB-D 角度通常落入 \[-30, 30\]（度）的范围；并放缩输入点云，对于 SUN RGB-D 比例通常落入 \[0.85, 1.15\] 的范围；最后平移输入点云，对于 SUN RGB-D 通常位移量为 0（即不做位移）。
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
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','img', 'gt_bboxes', 'gt_bboxes_labels'])
]
```

图像上的数据增强

- `Resize`：改变输入图像的大小，`keep_ratio=True` 意味着图像的比例不改变。
- `RandomFlip`：随机地翻折图像。

图像增强的实现取自 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/mmdet/datasets/transforms)。

## 度量指标

与 ScanNet 一样，通常使用 mAP（全类平均精度）来评估 SUN RGB-D 的检测任务的性能，比如 `mAP@0.25` 和 `mAP@0.5`。具体来说，评估时调用一个通用的计算 3D 物体检测多个类别的精度和召回率的函数。更多细节请参考 [`indoor_eval.py`](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/evaluation/functional/indoor_eval.py)。

因为 SUN RGB-D 包含有图像数据，所以图像上的物体检测也是可行的。举个例子，在 ImVoteNet 中，我们首先训练了一个图像检测器，并且也使用 mAP 指标，如 `mAP@0.5`，来评估其表现。我们使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 库中的 `eval_map` 函数来计算 mAP。
