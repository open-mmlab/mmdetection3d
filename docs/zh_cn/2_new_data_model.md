# 2: 在自定义数据集上进行训练

本文将主要介绍如何使用自定义数据集来进行模型的训练和测试，以 Waymo 数据集作为示例来说明整个流程。

基本步骤如下所示：

1. 准备自定义数据集；
2. 准备配置文件；
3. 在自定义数据集上进行模型的训练、测试和推理。

## 准备自定义数据集

在 MMDetection3D 中有三种方式来自定义一个新的数据集：

1. 将新数据集的数据格式重新组织成已支持的数据集格式；
2. 将新数据集的数据格式重新组织成已支持的一种中间格式；
3. 从头开始创建一个新的数据集。

由于前两种方式比第三种方式更加容易，我们更加建议采用前两种方式来自定义数据集。

在本文中，我们采用第一种方式来将 Waymo 数据集重新组织成 KITTI 数据集的数据格式。

**注意**：考虑到 Waymo 数据集的格式与现有的其他数据集的格式的差别较大，因此本文以该数据集为例来讲解如何自定义数据集，从而方便理解数据集自定义的过程。若需要创建的新数据集与现有的数据集的组织格式较为相似，如 Lyft 数据集和 nuScenes 数据集，采用对数据集的中间格式进行转换的方式（第二种方式）相比于采用对数据格式进行转换的方式（第一种方式）会更加简单易行。

### KITTI 数据集格式

应用于 3D 目标检测的 KITTI 原始数据集的组织方式通常如下所示，其中 `ImageSets` 包含数据集划分文件，用以划分训练集/验证集/测试集，`calib` 包含对于每个数据样本的标定信息，`image_2` 和 `velodyne` 分别包含图像数据和点云数据，`label_2` 包含与 3D 目标检测相关的标注文件。

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
```

KITTI 官方提供的目标检测开发[工具包](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)详细描述了 KITTI 数据集的标注格式，例如，KITTI 标注格式包含了以下的标注信息：

```
#  值    名称      描述
----------------------------------------------------------------------------
   1    类型      描述检测目标的类型：'Car'，'Van'，'Truck'，
                  'Pedestrian'，'Person_sitting'，'Cyclist'，'Tram'，
                  'Misc' 或 'DontCare'
   1    截断程度　 从 0（非截断）到 1（截断）的浮点数，其中截断指的是离开检测图像边界的检测目标
   1    遮挡程度　 用来表示遮挡状态的四种整数（0，1，2，3）:
                  0 = 可见，1 = 部分遮挡
                  2 = 大面积遮挡，3 = 未知
   1    观测角    观测目标的角度，取值范围为 [-pi..pi]
   4    标注框    检测目标在图像中的二维标注框（以0为初始下标）：包括每个检测目标的左上角和右下角的坐标
   3    维度　    检测目标的三维维度：高度、宽度、长度（以米为单位）
   3    位置　    相机坐标系下的三维位置 x，y，z（以米为单位）
   1    y 旋转　  相机坐标系下检测目标绕着Y轴的旋转角，取值范围为 [-pi..pi]
   1    得分　    仅在计算结果时使用，检测中表示置信度的浮点数，用于生成 p/r 曲线，在p/r 图中，越高的曲线表示结果越好。
```

接下来本文将对 Waymo 数据集原始格式进行转换。
首先需要将下载的 Waymo 数据集的数据文件和标注文件转换到 KITTI 数据集的格式，接着定义一个从 KittiDataset 类继承而来的 WaymoDataset 类，来帮助数据的加载、模型的训练和评估。

具体来说，首先使用[数据转换器](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/waymo_converter.py)将 Waymo 数据集转换成 KITTI 数据集的格式，并定义 [Waymo 类](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/waymo_dataset.py)对转换的数据进行处理。因为我们将 Waymo 原始数据集进行预处理并重新组织成 KITTI 数据集的格式，因此可以比较容易通过继承 KittiDataset 类来实现 WaymoDataset 类。需要注意的是，由于 Waymo 数据集有相应的官方评估方法，我们需要在定义新数据类的过程中引入官方评估方法，此时用户可以顺利的转换 Waymo 数据的格式，并使用 `WaymoDataset` 数据类进行模型的训练和评估。

更多关于 Waymo 数据集预处理的中间结果的细节，请参照对应的[说明文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/datasets/waymo_det.html)。


## 准备配置文件

第二步是准备配置文件来帮助数据集的读取和使用，另外，为了在 3D 检测中获得不错的性能，调整超参数通常是必要的。

假设我们想要使用 PointPillars 模型在 Waymo 数据集上实现三类的 3D 目标检测：vehicle、cyclist、pedestrian，参照 KITTI 数据集[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/kitti-3d-3class.py)、模型[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/hv_pointpillars_secfpn_kitti.py)和[整体配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py)，我们需要准备[数据集配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/waymoD5-3d-3class.py)、[模型配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/hv_pointpillars_secfpn_waymo.py))，并将这两种文件进行结合得到[整体配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)。

## 训练一个新的模型

为了使用一个新的配置文件来训练模型，可以通过下面的命令来实现：

```shell
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py
```

更多的使用细节，请参考[案例 1](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html)。

## 测试和推理

为了测试已经训练好的模型的性能，可以通过下面的命令来实现：

```shell
python tools/test.py configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py work_dirs/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/latest.pth --eval waymo
```

**注意**：为了使用 Waymo 数据集的评估方法，需要参考[说明文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/datasets/waymo_det.html)并按照官方指导来准备与评估相关联的文件。

更多有关测试和推理的使用细节，请参考[案例 1](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html) 。
