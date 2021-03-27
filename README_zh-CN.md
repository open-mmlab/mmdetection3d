<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)


**新闻**: 我们发布了版本v0.11.0.

在第三届[ nuScenes 3D 检测挑战赛](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)（第五届 AI Driving Olympics, NeurIPS 2020）中，我们获得了最佳 PKL 奖、第三名和最好的纯视觉的结果，相关的代码和模型将会在不久后发布。

文档: https://mmdetection3d.readthedocs.io/

## 简介

[English](README.md) | 简体中文

主分支代码目前支持 PyTorch 1.3 以上的版本。

MMDetection3D 是一个基于 PyTorch 的目标检测开源工具箱, 下一代面向3D检测的平台. 它是 OpenMMlab 项目的一部分，这个项目由香港中文大学多媒体实验室和商汤科技联合发起.

![demo image](resources/mmdet3d_outdoor_demo.gif)

### 主要特性

- **支持多模态/单模态的检测器**

  支持多模态/单模态检测器，包括 MVXNet，VoteNet，PointPillars 等。

- **支持户内/户外的数据集**

  支持室内/室外的3D检测数据集，包括 ScanNet, SUNRGB-D, Waymo, nuScenes, Lyft, KITTI.

  对于 nuScenes 数据集, 我们也支持 [nuImages 数据集](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages).

- **与 2D 检测器的自然整合**

   [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) 支持的**300+个模型 , 40+的论文算法**, 和相关模块都可以在此代码库中训练或使用。

- **性能高**

   训练速度比其他代码库更快。下表可见主要的对比结果。更多的细节可见[基准测评文档](./docs/benchmarks.md)。我们对比了每秒训练的样本数（值越高越好）。其他代码库不支持的模型被标记为 `×`。

  | Methods | MMDetection3D | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) |[votenet](https://github.com/facebookresearch/votenet)| [Det3D](https://github.com/poodarchu/Det3D) |
  |:-------:|:-------------:|:---------:|:-----:|:-----:|
  | VoteNet | 358           | ×         |   77  | ×     |
  | PointPillars-car| 141           | ×         |   ×  | 140     |
  | PointPillars-3class| 107           |44     |   ×      | ×    |
  | SECOND| 40           |30     |   ×      | ×    |
  | Part-A2| 17           |14     |   ×      | ×    |

和 [MMDetection](https://github.com/open-mmlab/mmdetection)，[MMCV](https://github.com/open-mmlab/mmcv) 一样, MMDetection3D 也可以作为一个库去支持各式各样的项目.

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 更新日志

最新的版本 v0.11.0 在 2021.03.01发布。
如果想了解更多版本更新细节和历史信息，请阅读[更新日志](docs/changelog.md)。

## 基准测试和模型库

测试结果和模型可以在[模型库](docs/model_zoo.md)中找到。

已支持的骨干网络：

- [x] PointNet (CVPR'2017)
- [x] PointNet++ (NeurIPS'2017)
- [x] RegNet (CVPR'2020)

已支持的算法：

- [x] [SECOND (Sensor'2018)](configs/second/README.md)
- [x] [PointPillars (CVPR'2019)](configs/pointpillars/README.md)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor/README.md)
- [x] [VoteNet (ICCV'2019)](configs/votenet/README.md)
- [x] [H3DNet (ECCV'2020)](configs/h3dnet/README.md)
- [x] [3DSSD (CVPR'2020)](configs/3dssd/README.md)
- [x] [Part-A2 (TPAMI'2020)](configs/parta2/README.md)
- [x] [MVXNet (ICRA'2019)](configs/mvxnet/README.md)
- [x] [CenterPoint (CVPR'2021)](configs/centerpoint/README.md)
- [x] [SSN (ECCV'2020)](configs/ssn/README.md)
- [x] [ImVoteNet (CVPR'2020)](configs/imvotenet/README.md)

|                    | ResNet   | ResNeXt  | SENet    |PointNet++ | HRNet | RegNetX | Res2Net |
|--------------------|:--------:|:--------:|:--------:|:---------:|:-----:|:--------:|:-----:|
| SECOND             | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| PointPillars       | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| FreeAnchor         | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| VoteNet            | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |
| H3DNet            | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |
| 3DSSD            | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |
| Part-A2            | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| MVXNet             | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| CenterPoint        | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| SSN                | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| ImVoteNet            | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |

其他特性
- [x] [Dynamic Voxelization](configs/dynamic_voxelization/README.md)

**注意：** [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) 支持的基于2D检测的**300+个模型 , 40+的论文算法**在 MMDetection3D 中都可以被训练或使用。

## 安装

请参考[快速入门文档](docs/get_started.md)进行安装。

## 快速入门

请参考[快速入门文档](docs/get_started.md)学习 MMDetection3D 的基本使用。 我们为新手提供了分别针对[已有数据集](docs/1_exist_data_model.md)和[新数据集](docs/2_new_data_model.md)的使用指南。我们也提供了一些进阶教程，内容覆盖了[学习配置文件](docs/tutorials/config.md), [增加数据集支持](docs/tutorials/customize_dataset.md), [设计新的数据预处理流程](docs/tutorials/data_pipeline.md), [增加自定义模型](docs/tutorials/customize_models.md), [增加自定义的运行时配置](docs/tutorials/customize_runtime.md)和 [Waymo 数据集](docs/tutorials/waymo.md).

## 引用

如果你觉得本项目对你的研究工作有所帮助，请参考如下 bibtex 引用 MMdetection3D

```latex
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMDetection3D 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMDetection3D 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新的 3D 检测模型。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
