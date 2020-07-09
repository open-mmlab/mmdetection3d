<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
</div>

**News**: We released the codebase v0.1.0.

Documentation: https://mmdetection3d.readthedocs.io/

## Introduction

The master branch works with **PyTorch 1.3 to 1.5**.

MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is
a part of the OpenMMLab project developed by [MMLab](http://mmlab.ie.cuhk.edu.hk/).

![demo image](resources/outdoor_demo.gif)

### Major features

- **Support multi-modality/single-modality detectors out of box**

  It directly supports multi-modality/single-modality detectors including MVXNet, VoteNet, PointPillars, etc.

- **Support indoor/outdoor 3D detection out of box**

  It directly supports popular indoor and outdoor 3D detection datasets, including ScanNet, SUNRGB-D, nuScenes, Lyft, and KITTI.

- **Natural integration with 2D detection**

  All the about **300 models, methods of 40+ papers**, and modules supported in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

- **High efficiency**

  It trains [faster than other codebases](./docs/benchmarks.md).

Apart from MMDetection3D, we also released a library [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv) for computer vision research, which are heavily depended on by this toolbox. Like [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv), MMDetection3D can also be used as a library to support different projects on top of it.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.1.0 was released in 9/7/2020.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [model zoo](docs/model_zoo.md).

|                    | ResNet   | ResNeXt  | SENet    |PointNet++ | HRNet | RegNetX | Res2Net |
|--------------------|:--------:|:--------:|:--------:|:---------:|:-----:|:--------:|:-----:|
| SECOND             | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| PointPillars       | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| FreeAnchor         | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| VoteNet            | ✗        | ✗        | ✗        | ✓         | ✗     | ✗        | ✗     |
| Part-A2            | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |
| MVXNet             | ☐        | ☐        | ☐        | ✗         | ☐     | ✓        | ☐     |

Other features
- [x] [Dynamic Voxelization](configs/carafe/README.md)

**Note:** All the about **300 models, methods of 40+ papers** in 2D detection supported by [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Contributing

We appreciate all contributions to improve MMDetection3D. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection3D is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new 3D detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{mmdetection3d_2020,
  title   = {{MMDetection3D}},
  author  = {Zhang, Wenwei and Wu, Yuefeng and Wang, Tai and Li, Yinhao and
             Lin, Kwan-Yee and Wang, Zhe and Shi, Jianping and Qian, Chen and
             Chen, Kai, and Lin, Dahua and Loy, Chen Change},
  howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
  year =         {2020}
}
```

## Contact

This repo is currently maintained by Wenwei Zhang ([@ZwwWayne](https://github.com/ZwwWayne)), Yuefeng Wu ([@xavierwu95](https://github.com/xavierwu95)), Tai Wang ([@Tai-Wang](https://github.com/Tai-Wang)), and Yinhao Li ([@yinchimaoliang](https://github.com/yinchimaoliang)).
