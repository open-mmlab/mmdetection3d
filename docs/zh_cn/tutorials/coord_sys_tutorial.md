# 教程 6: 坐标系

## Overview概述

MMDetection3D 使用 3 种不同的坐标系。3D 目标检测领域中不同坐标系的存在是有必要的，因为对于不同的 3D 数据采集设备，如激光雷达、深度相机等，其坐标系并不一致，因此不同的 3D 数据集也遵循不同的数据格式。早期的工作，如 SECOND、VoteNet 将原始数据转换为另一种格式，形成了一些后续工作也遵循的约定，使得不同坐标系之间的转换变得更加复杂。

尽管数据集和采集设备多种多样，但是通过总结 3D 目标检测的工作线，我们可以将坐标系大致分为三类：

- 相机坐标系 -- 大多数相机的坐标系，其中 y 轴正方向指向地面，x 轴正方向指向右侧，z 轴正方向指向前方。
    ```
               up  z front
                |    ^
                |   /
                |  /
                | /
                |/
    left ------ 0 ------> x right
                |
                |
                |
                |
                v
              y down
    ```
- 激光雷达坐标系 -- 大多数激光雷达的坐标系，其中 z 轴负方向指向地面，x 轴正方向指向前方，y 轴正方向指向左侧。
    ```
                 z up  x front
                   ^    ^
                   |   /
                   |  /
                   | /
                   |/
    y left <------ 0 ------ right
    ```
- 深度坐标系 -- VoteNet、H3DNet 等使用的坐标系，其中 z 轴负方向指向地面，x 轴正方向指向右侧，y 轴正方向指向前方。
    ```
              z up  y front
                ^    ^
                |   /
                |  /
                | /
                |/
    left ------ 0 ------> x right
    ```

该教程中的坐标系定义实际上**不仅仅是定义三个轴**。对于形如 ``$$`(x, y, z, dx, dy, dz, r)`$$`` 的框，我们的坐标系也定义了如何解释框的尺寸 ``$$`(dx, dy, dz)`$$`` 和转向角 ``$$`r`$$``。

三个坐标系的图示如下：

![](https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v1.0.0.dev0/resources/coord_sys_all.png)

上面三张图是 3D 坐标系，下面三张图是鸟瞰图。

以后我们将坚持使用本教程中定义的三个坐标系。

## 转向角的定义

请参考[维基百科](https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles)了解转向角的标准定义。在目标检测中，我们选择一个轴作为重力轴，并在垂直于重力轴的平面 ``$$`\Pi`$$`` 上选取一个参考方向，那么参考方向的转向角为 0，在 ``$$`\Pi`$$`` 上的其他方向有非零的转向角，这取决于其与参考方向的角度。

目前，对于所有支持的数据集，标注不包括俯仰角和滚动角，这意味着我们在预测框和计算框之间的重叠时只需考虑转向角。

在 MMDetection3D 中，所有坐标系都是右手坐标系，这意味着如果从重力轴的负方向（轴的正方向指向人眼）看，转向角沿着逆时针方向增加。

下图显示，在右手坐标系中，如果我们设置 x 轴正方向为参考方向，那么 y 轴正方小的转向角为 ``$$`\frac{\pi}{2}`$$``。

```
                     z up  y front (yaw=0.5*pi)
                      ^    ^
                      |   /
                      |  /
                      | /
                      |/
left (yaw=pi)  ------ 0 ------> x right (yaw=0)
```

对于一个框来说，其转向角的值等于其方向减去一个参考方向。在 MMDetection3D 的所有坐标系中，参考方向总是 x 轴正方向，而如果一个框的转向角为 0，则其方向被定义为与 x 轴平行。框的转向角的定义如下图所示。

```
y front
  ^      box direction (yaw=0.5*pi)
 /|\        ^
  |        /|\
  |     ____|____
  |    |    |    |
  |    |    |    |
__|____|____|____|______\ x right
  |    |    |    |      /
  |    |    |    |
  |    |____|____|
  |
```

## 框尺寸的定义

框尺寸的定义与转向角的定义是分不开的。在上一节中，我们提到如果一个框的转向角为 0，它的方向就被定义为与 x 轴平行。那么自然地，与 x 轴相对应的框的尺寸应该是 ``$$`dx`$$``。但是，这在某些数据集中并非总是这样的（我们将会解决这个问题）。

下图展示了 x 轴和 ``$$`dx`$$``，y 轴和 ``$$`dy`$$`` 对应的含义。

```
y front
  ^      box direction (yaw=0.5*pi)
 /|\        ^
  |        /|\
  |     ____|____
  |    |    |    |
  |    |    |    | dx
__|____|____|____|______\ x right
  |    |    |    |      /
  |    |    |    |
  |    |____|____|
  |         dy
```

注意框的方向总是和 ``$$`dx`$$`` 边平行。

```
y front
  ^     _________
 /|\   |    |    |
  |    |    |    |
  |    |    |    | dy
  |    |____|____|____\  box direction (yaw=0)
  |    |    |    |    /
__|____|____|____|_________\ x right
  |    |    |    |         /
  |    |____|____|
  |         dx
  |
```

## 与支持的数据集的原始坐标系的关系

### KITTI

KITTI 数据集的原始标注是在相机坐标系下的，详见 [get_label_anno](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/tools/data_converter/kitti_data_utils.py)。在 MMDetection3D 中，为了在 KITTI 数据集上训练基于激光雷达的模型，首先将数据从相机坐标系转换到激光雷达坐标系，详见 [get_ann_info](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/datasets/kitti_dataset.py)。为了训练基于视觉的模型，数据保持在相机坐标系下不变。

在 SECOND 中，一个框的激光雷达坐标系定义如下（鸟瞰图）：

![](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

对于每个框，其尺寸为 ``$$`(w, l, h)`$$``，转向角的参考方向为 y 轴正方向。更多细节请参考[代码库](https://github.com/traveller59/second.pytorch#concepts)。

我们的激光雷达坐标系有两处改变：

- 为了保持一致性，转向角被定义为右手而非左手；
- 框的尺寸为 ``$$`(l, w, h)`$$`` 而非 ``$$`(w, l, h)`$$``，因为在 KITTI 中 ``$$`w`$$`` 对应 ``$$`dy`$$``，``$$`l`$$`` 对应 ``$$`dx`$$``。

### Waymo

我们使用 Waymo 数据集的 KITTI 格式数据。因此，在我们的实现中 KITTI 和 Waymo 也共用相同的坐标系。

### NuScenes

NuScenes 为评估提供了一个工具包，其中每个框都被包装成一个 `Box` 实例。`Box` 的坐标系不同于我们的激光雷达坐标系，在 `Box` 坐标系中，前两个表示框尺寸的元素分别对应 ``$$`(dy, dx)`$$`` 或 ``$$`(w, l)`$$``，这和我们的表示相反。更多细节请参考 NuScenes [教程](https://github.com/open-mmlab/mmdetection3d/blob/dev/docs/zh_cn/datasets/nuscenes_det.md#notes)。

读者可以参考 [NuScenes 开发工具](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/detection)，了解 [NuScenes 框]((https://github.com/nutonomy/nuscenes-devkit/blob/2c6a752319f23910d5f55cc995abc547a9e54142/python-sdk/nuscenes/utils/data_classes.py#L457)) 的定义和 [NuScenes 评估](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py)的实现。

### Lyft

就涉及坐标系而言，Lyft 和 NuScenes 共享相同的数据格式。

请参考[官方网站](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data)获取更多信息。

### ScanNet

ScanNet 的原始数据不是点云而是网格。采样的点云数据是在我们的深度坐标系下获得的。对于 ScanNet 检测任务，框的标注是轴对齐的，并且转向角始终是 0。因此，我们的深度坐标系中转向角的定义对 ScanNet 没有影响。

### SUN RGB-D

SUN RGB-D 的原始数据不是点云，而是 RGB-D 图像。我们通过反投影为每张图像得到对应的点云，其在我们的深度坐标系下。然而，标注并不在我们的坐标系下，因此需要转换。

为了将原始标注转换到我们的深度坐标系，请参考 [sunrgbd_data_utils.py](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/tools/data_converter/sunrgbd_data_utils.py)。

### S3DIS

在我们的实现中，S3DIS 与 ScanNet 共享相同的坐标系。然而 S3DIS 是一个仅限于分割任务的数据集，因此没有标注是坐标系敏感的。

## 例子

### 框（在不同坐标系间）的转换

以相机坐标系和激光雷达坐标系间的转换为例：

首先，对于点和框的中心点，坐标转换前后满足下列关系：

- ``$$`x_{LiDAR}=z_{camera}`$$``
- ``$$`y_{LiDAR}=-x_{camera}`$$``
- ``$$`z_{LiDAR}=-y_{camera}`$$``

然后，框的尺寸转换前后满足下列关系：

- ``$$`dx_{LiDAR}=dx_{camera}`$$``
- ``$$`dy_{LiDAR}=dz_{camera}`$$``
- ``$$`dz_{LiDAR}=dy_{camera}`$$``

最后，转向角也应该被转换：

- ``$$`r_{LiDAR}=-\frac{\pi}{2}-r_{camera}`$$``

详见[此处](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/box_3d_mode.py)代码了解更多细节。

### 鸟瞰图

如果 3D 框是 ``$$`(x, y, z, dx, dy, dz, r)`$$``，相机坐标系下框的鸟瞰图是 ``$$`(x, z, dx, dz, -r)`$$``。转向角符号取反是因为相机坐标系重力轴的正方向指向地面。

详见[此处](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/cam_box3d.py)代码了解更多细节。

### 框的旋转

我们将各种框的旋转设定为绕着重力轴逆时针旋转。因此，为了旋转一个 3D 框，我们首先计算新框的中心，然后将旋转角度添加到转向角。

详见[此处](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/cam_box3d.py)代码了解更多细节。

## 常见问题

#### Q1: 与框相关的算子是否适用于所有坐标系类型？

否。例如，在该[文件夹](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/ops/roiaware_pool3d)下的算子只适用于深度或激光雷达坐标系。由于如果从上方看，旋转是顺时针的，所以 KITTI 数据集[这里](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/evaluation/kitti_utils)的评估函数仅适用于相机坐标系。

对于每个和框相关的算子，我们注明了该算子所适用的框类型。

#### Q2: 在每个坐标系中，三个轴是否分别指向右侧、前方和地面？

否。例如在 KITTI 中，当从相机坐标系转换为激光雷达坐标系时，我们需要一个校准矩阵。

#### Q3: 框中转向角 ``$$`2\pi`$$`` 的相位差如何影响评估？

对于 IoU 计算，转向角有 ``$$`2\pi`$$`` 的相位差会导致相同的框，因此不会影响评估。

对于角度预测评估，例如 NuScenes 中的 NDS 指标和 KITTI 中的 AOS 指标，预测框的角度首先会被标准化，因此 ``$$`2\pi`$$`` 的相位差不会改变结果。

#### Q4: 框中转向角 ``$$`\pi`$$`` 的相位差如何影响评估？

对于 IoU 计算，转向角有 ``$$`\pi`$$`` 的相位差会导致相同的框，因此不会影响评估。

然而，对于角度预测评估，这会导致完全相反的方向。

考虑一辆汽车，转向角是汽车前部方向与 x 轴正方向之间的角度。如果我们在该角度加上 ``$$`\pi`$$``，汽车前部将变成汽车尾部。

对于像障碍物这样的类别，前部和尾部没有区别，因此 ``$$`\pi`$$`` 的相位差不会影响角度预测得分。
