<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/1.1/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)

</div>

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

**News**:

**v1.1.0rc3** was released in 7/1/2023

The compatibilities of models are broken due to the unification and simplification of coordinate systems after v1.0.0rc0. For now, most models are benchmarked with similar performance, though few models are still being benchmarked. In the following release, we will update all the model checkpoints and benchmarks. See more details in the [Changelog](docs/en/notes/changelog.md) and [Changelog-v1.0.x](docs/en/notes/changelog_v1.0.x.md).

Documentation: https://mmdetection3d.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

The master branch works with **PyTorch 1.6+**.

MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is
a part of the OpenMMLab project developed by [MMLab](http://mmlab.ie.cuhk.edu.hk/).

![demo image](resources/mmdet3d_outdoor_demo.gif)

### Major features

- **Support multi-modality/single-modality detectors out of box**

  It directly supports multi-modality/single-modality detectors including MVXNet, VoteNet, PointPillars, etc.

- **Support indoor/outdoor 3D detection out of box**

  It directly supports popular indoor and outdoor 3D detection datasets, including ScanNet, SUNRGB-D, Waymo, nuScenes, Lyft, and KITTI.
  For nuScenes dataset, we also support [nuImages dataset](https://github.com/open-mmlab/mmdetection3d/tree/1.1/configs/nuimages).

- **Natural integration with 2D detection**

  All the about **300+ models, methods of 40+ papers**, and modules supported in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/model_zoo.md) can be trained or used in this codebase.

- **High efficiency**

  It trains faster than other codebases. The main results are as below. Details can be found in [benchmark.md](./docs/en/notes/benchmarks.md). We compare the number of samples trained per second (the higher, the better). The models that are not supported by other codebases are marked by `✗`.

  |       Methods       | MMDetection3D | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | [votenet](https://github.com/facebookresearch/votenet) | [Det3D](https://github.com/poodarchu/Det3D) |
  | :-----------------: | :-----------: | :--------------------------------------------------: | :----------------------------------------------------: | :-----------------------------------------: |
  |       VoteNet       |      358      |                          ✗                           |                           77                           |                      ✗                      |
  |  PointPillars-car   |      141      |                          ✗                           |                           ✗                            |                     140                     |
  | PointPillars-3class |      107      |                          44                          |                           ✗                            |                      ✗                      |
  |       SECOND        |      40       |                          30                          |                           ✗                            |                      ✗                      |
  |       Part-A2       |      17       |                          14                          |                           ✗                            |                      ✗                      |

Like [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv), MMDetection3D can also be used as a library to support different projects on top of it.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

**1.1.0rc3** was released in 7/1/2023.

Please refer to [changelog.md](docs/en/notes/changelog.md) for details and release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Heads</b>
      </td>
      <td>
        <b>Features</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li><a href="configs/pointnet2">PointNet (CVPR'2017)</a></li>
        <li><a href="configs/pointnet2">PointNet++ (NeurIPS'2017)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/dgcnn">DGCNN (TOG'2019)</a></li>
        <li>DLA (CVPR'2018)</li>
        <li>MinkResNet (CVPR'2019)</li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/dynamic_voxelization">Dynamic Voxelization (CoRL'2019)</a></li>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="middle">
      <td>
        <b>3D Object Detection</b>
      </td>
      <td>
        <b>Monocular 3D Object Detection</b>
      </td>
      <td>
        <b>Multi-modal 3D Object Detection</b>
      </td>
      <td>
        <b>3D Semantic Segmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <li><b>Outdoor</b></li>
        <ul>
            <li><a href="configs/second">SECOND (Sensor'2018)</a></li>
            <li><a href="configs/pointpillars">PointPillars (CVPR'2019)</a></li>
            <li><a href="configs/ssn">SSN (ECCV'2020)</a></li>
            <li><a href="configs/3dssd">3DSSD (CVPR'2020)</a></li>
            <li><a href="configs/sassd">SA-SSD (CVPR'2020)</a></li>
            <li><a href="configs/point_rcnn">PointRCNN (CVPR'2019)</a></li>
            <li><a href="configs/parta2">Part-A2 (TPAMI'2020)</a></li>
            <li><a href="configs/centerpoint">CenterPoint (CVPR'2021)</a></li>
            <li><a href="configs/pv_rcnn">PV-RCNN (CVPR'2020)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
            <li><a href="configs/votenet">VoteNet (ICCV'2019)</a></li>
            <li><a href="configs/h3dnet">H3DNet (ECCV'2020)</a></li>
            <li><a href="configs/groupfree3d">Group-Free-3D (ICCV'2021)</a></li>
            <li><a href="configs/fcaf3d">FCAF3D (ECCV'2022)</a></li>
      </ul>
      </td>
      <td>
        <li><b>Outdoor</b></li>
        <ul>
          <li><a href="configs/imvoxelnet">ImVoxelNet (WACV'2022)</a></li>
          <li><a href="configs/smoke">SMOKE (CVPRW'2020)</a></li>
          <li><a href="configs/fcos3d">FCOS3D (ICCVW'2021)</a></li>
          <li><a href="configs/pgd">PGD (CoRL'2021)</a></li>
          <li><a href="configs/monoflex">MonoFlex (CVPR'2021)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
          <li><a href="configs/imvoxelnet">ImVoxelNet (WACV'2022)</a></li>
        </ul>
      </td>
      <td>
        <li><b>Outdoor</b></li>
        <ul>
          <li><a href="configs/mvxnet">MVXNet (ICRA'2019)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
          <li><a href="configs/imvotenet">ImVoteNet (CVPR'2020)</a></li>
        </ul>
      </td>
      <td>
        <li><b>Indoor</b></li>
        <ul>
          <li><a href="configs/pointnet2">PointNet++ (NeurIPS'2017)</a></li>
          <li><a href="configs/paconv">PAConv (CVPR'2021)</a></li>
          <li><a href="configs/dgcnn">DGCNN (TOG'2019)</a></li>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

|               | ResNet | PointNet++ | SECOND | DGCNN | RegNetX | DLA | MinkResNet |
| :-----------: | :----: | :--------: | :----: | :---: | :-----: | :-: | :--------: |
|    SECOND     |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |
| PointPillars  |   ✗    |     ✗      |   ✓    |   ✗   |    ✓    |  ✗  |     ✗      |
|  FreeAnchor   |   ✗    |     ✗      |   ✗    |   ✗   |    ✓    |  ✗  |     ✗      |
|    VoteNet    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|    H3DNet     |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|     3DSSD     |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|    Part-A2    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |
|    MVXNet     |   ✓    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |
|  CenterPoint  |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |
|      SSN      |   ✗    |     ✗      |   ✗    |   ✗   |    ✓    |  ✗  |     ✗      |
|   ImVoteNet   |   ✓    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|    FCOS3D     |   ✓    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|  PointNet++   |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
| Group-Free-3D |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|  ImVoxelNet   |   ✓    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|    PAConv     |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|     DGCNN     |   ✗    |     ✗      |   ✗    |   ✓   |    ✗    |  ✗  |     ✗      |
|     SMOKE     |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✓  |     ✗      |
|      PGD      |   ✓    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |
|   MonoFlex    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✓  |     ✗      |
|    SA-SSD     |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |
|    FCAF3D     |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✓      |
|    PV-RCNN    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |

**Note:** All the about **300+ models, methods of 40+ papers** in 2D detection supported by [MMDetection](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/model_zoo.md) can be trained or used in this codebase.

## Installation

Please refer to [getting_started.md](docs/en/getting_started.md) for installation.

## Get Started

Please see [getting_started.md](docs/en/getting_started.md) for the basic usage of MMDetection3D. We provide guidance for quick run [with existing dataset](docs/en/user_guides/train_test.md) and [with new dataset](docs/en/user_guides/2_new_data_model.md) for beginners. There are also tutorials for [learning configuration systems](docs/en/user_guides/config.md), [customizing dataset](docs/en/advanced_guides/customize_dataset.md), [designing data pipeline](docs/en/user_guides/data_pipeline.md), [customizing models](docs/en/advanced_guides/customize_models.md), [customizing runtime settings](docs/en/advanced_guides/customize_runtime.md) and [Waymo dataset](docs/en/advanced_guides/datasets/waymo_det.md).

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions. When updating the version of MMDetection3D, please also check the [compatibility doc](docs/en/notes/compatibility.md) to be aware of the BC-breaking updates introduced in each version.

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

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMEval](https://github.com/open-mmlab/mmeval): A unified evaluation library for multiple machine learning libraries.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
