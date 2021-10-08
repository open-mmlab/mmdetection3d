<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)


**News**: We released the codebase v0.17.1.

We are going through large refactoring to provide simpler and more unified usage of many modules. Thus, few features will be added to the master branch in the following months.

The compatibilities of models are broken due to the unification and simplification of coordinate systems. For now, most models are benchmarked with similar performance, though few models are still being benchmarked.

You can start experiments with v1.0.0.dev0 if you are interested. Please note that our new features will only be supported in v1.0.0 branch afterward.

In the [nuScenes 3D detection challenge](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) of the 5th AI Driving Olympics in NeurIPS 2020, we obtained the best PKL award and the second runner-up by multi-modality entry, and the best vision-only results.

Code and models for the best vision-only method, [FCOS3D](https://arxiv.org/abs/2104.10956), have been released. Please stay tuned for [MoCa](https://arxiv.org/abs/2012.12741).

Documentation: https://mmdetection3d.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

The master branch works with **PyTorch 1.3+**.

MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is
a part of the OpenMMLab project developed by [MMLab](http://mmlab.ie.cuhk.edu.hk/).

![demo image](resources/mmdet3d_outdoor_demo.gif)

### Major features

- **Support multi-modality/single-modality detectors out of box**

  It directly supports multi-modality/single-modality detectors including MVXNet, VoteNet, PointPillars, etc.

- **Support indoor/outdoor 3D detection out of box**

  It directly supports popular indoor and outdoor 3D detection datasets, including ScanNet, SUNRGB-D, Waymo, nuScenes, Lyft, and KITTI.
  For nuScenes dataset, we also support [nuImages dataset](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages).

- **Natural integration with 2D detection**

  All the about **300+ models, methods of 40+ papers**, and modules supported in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

- **High efficiency**

  It trains faster than other codebases. The main results are as below. Details can be found in [benchmark.md](./docs/benchmarks.md). We compare the number of samples trained per second (the higher, the better). The models that are not supported by other codebases are marked by `×`.

  | Methods | MMDetection3D | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) |[votenet](https://github.com/facebookresearch/votenet)| [Det3D](https://github.com/poodarchu/Det3D) |
  |:-------:|:-------------:|:---------:|:-----:|:-----:|
  | VoteNet | 358           | ×         |   77  | ×     |
  | PointPillars-car| 141           | ×         |   ×  | 140     |
  | PointPillars-3class| 107           |44     |   ×      | ×    |
  | SECOND| 40           |30     |   ×      | ×    |
  | Part-A2| 17           |14     |   ×      | ×    |

Like [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv), MMDetection3D can also be used as a library to support different projects on top of it.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.17.1 was released in 1/10/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

For branch v1.0.0.dev0, please refer to [changelog_v1.0.md](https://github.com/Tai-Wang/mmdetection3d/blob/v1.0.0.dev0-changelog/docs/changelog_v1.0.md) for our latest features and more details.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [model zoo](docs/model_zoo.md).

Support backbones:

- [x] PointNet (CVPR'2017)
- [x] PointNet++ (NeurIPS'2017)
- [x] RegNet (CVPR'2020)

Support methods

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
- [x] [FCOS3D (Arxiv'2021)](configs/fcos3d/README.md)
- [x] [PointNet++ (NeurIPS'2017)](configs/pointnet2/README.md)
- [x] [Group-Free-3D (Arxiv'2021)](configs/groupfree3d/README.md)
- [x] [ImVoxelNet (Arxiv'2021)](configs/imvoxelnet/README.md)
- [x] [PAConv (CVPR'2021)](configs/paconv/README.md)

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
| FCOS3D               | ✓        | ☐        | ☐        | ✗         | ☐     | ☐        | ☐     |
| PointNet++           | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |
| Group-Free-3D        | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |
| ImVoxelNet           | ✓         | ✗        | ✗        | ✗        | ✗     | ✗        | ✗     |
| PAConv               | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |

Other features
- [x] [Dynamic Voxelization](configs/dynamic_voxelization/README.md)

**Note:** All the about **300+ models, methods of 40+ papers** in 2D detection supported by [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

## Installation

Please refer to [getting_started.md](docs/getting_started.md) for installation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection3D. We provide guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with customized dataset](docs/2_new_data_model.md) for beginners. There are also tutorials for [learning configuration systems](docs/tutorials/config.md), [adding new dataset](docs/tutorials/customize_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [Waymo dataset](docs/datasets/waymo_det.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions. When updating the version of MMDetection3D, please also check the [compatibility doc](docs/compatibility.md) to be aware of the BC-breaking updates introduced in each version.

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMDetection3D. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection3D is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new 3D detectors.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
