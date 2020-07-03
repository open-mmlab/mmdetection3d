# MMDetection3D

**News**: We released the codebase v0.1.0.

Documentation: https://mmdetection3d.readthedocs.io/

## Introduction

The master branch works with **PyTorch 1.3 to 1.5**.

MMDetection3D is an open source object detection toolbox based on PyTorch. It is
a part of the OpenMMLab project developed by [MMLab](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  The training speed is [faster than other codebases](./docs/benchmarks.md).

- **State of the art**

  The accuracy of models is [faster than other codebases](./docs/benchmarks.md).

Apart from MMDetection3D, we also released a library [MMDetection](https://github.com/open-mmlab/mmdetection) and [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which are heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.1.0 was released in 24/6/2020.
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

All the about **300 models, 40+ papers**, and modules supported in [MMDetection's model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.


## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection3D is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new 3D detectors.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{mmdetection3d_2020,
  title   = {{MMDetection3D}},
  author  = {Zhang, Wenwei and Wu, Yuefeng and Li, Yinhao and Lin, Kwan-Yee and
             Qian, Chen, Shi, Jianping, and Chen, Kai, and Li, Hongsheng and
             Lin, Dahua, and Loy, Chen Change},
  howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
  year =         {2020}
}
```


## Contact

This repo is currently maintained by Wenwei Zhang ([@ZwwWayne](https://github.com/ZwwWayne)).
