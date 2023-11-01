# NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection

> [NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection](https://arxiv.org/abs/2307.14620)

<!-- [ALGORITHM] -->

## Abstract

NeRF-Det is a novel method for indoor 3D detection with posed RGB images as input. Unlike existing indoor 3D detection methods that struggle to model scene geometry,NeRF-Det makes novel use of NeRF in an end-to-end manner to explicitly estimate 3D geometry, thereby improving 3D detection performance. Specifically, to avoid the significant extra latency associated with per-scene optimization of NeRF, NeRF-Det introduce sufficient geometry priors to enhance the generalizability of NeRF-MLP. Furthermore, it subtly connect the detection and NeRF branches through a shared MLP, enabling an efficient adaptation of NeRF to detection and yielding geometry-aware volumetric representations for 3D detection. NeRF-Det outperforms state-of-the-arts by 3.9 mAP and 3.1 mAP on the ScanNet and ARKITScenes benchmarks, respectively. The author provide extensive analysis to shed light on how NeRF-Det works. As a result of joint-training design, NeRF-Det is able to generalize well to unseen scenes for object detection, view synthesis, and depth estimation tasks without requiring per-scene optimization.Code will be available at https://github.com/facebookresearch/NeRF-Det

<div align=center>
<img src="https://chenfengxu714.github.io/nerfdet/static/images/method-cropped_1.png" width="800"/>
</div>

## Introduction

This directory contains the implementations of NeRF-Det (https://arxiv.org/abs/2307.14620). Our implementations are built on top of MMdetection3D.We have updated NeRF-Det to be compatible with latest mmdet3d version. The codebase and config files have all changed to adapt to the new mmdet3d version. All previous pretrained models are verified with the result listed below. However, newly trained models are yet to be uploaded.

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

## Dataset

The format of the scannet dataset in the latest version of mmdet3d only supports the lidar tasks.For NeRF-Det,we need to create the new format of scannet dataset.

To perpare the new dataset, please use the `update_infos_to_v2.py` in this folder to replace the file with the same name in `tools/dataset_converters/`.And then refer to [scannet dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/scannet.html) and following the instructions.

The new format of the pkl is organized as below:

- scannet_infos_train.pkl: The train data infos, the detailed info of each scan is as follows:
  - info\['instances'\]:A list of dict contains all annotations, each dict contains all annotation information of single instance.For the i-th instance:
    - info\['instances'\]\[i\]\['bbox_3d'\]: List of 6 numbers representing the axis_aligned in depth coordinate system, in (x,y,z,l,w,h) order.
    - info\['instances'\]\[i\]\['bbox_label_3d'\]: The label of each 3d bounding boxes.
  - info\['intrinsics'\]: The intrinsic matrix.Every scene has one matrix.
  - info\['extrinsics'\]: The extrinsic matrixes.Every scene has 300 matrixes.
  - info\['img_paths'\]: The paths of the 300 rgb pictures.
  - info\['axis_align_matrix'\]: The align matrix.Every scene has one matrix.

Also, you can download the processed pkls [here](<>).

After preparing your scannet dataset pkls,please change the paths in configs to fit your project.

## Train

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/NeRF-Det/configs/nerfdet_res50_2x_low_res.py ${WORK_DIR}
```

## Evaluation using pretrained models

1. Download the pretrained weights accordingly. (Here NeRF-Det-R50\* means this model uses depth in the training step)

   |                            Backbone                             | mAP@25 | mAP@50 |  Download   |    Log    |
   | :-------------------------------------------------------------: | :----: | :----: | :---------: | :-------: |
   |      [NeRF-Det-R50](./configs/nerfdet_res50_2x_low_res.py)      |  53.0  |  26.7  | [model](<>) | [log](<>) |
   |  [NeRF-Det-R50\*](./configs/nerfdet_res50_2x_low_res_depth.py)  |  52.4  |  29.1  | [model](<>) | [log](<>) |
   | [NeRF-Det-R101\*](./configs/nerfdet_res101_2x_low_res_depth.py) |  52.2  |  28.8  | [model](<>) | [log](<>) |

2. Testing

   To test, use:

   ```bash
   python tools/test.py projects/NeRF-Det/configs/nerfdet_res50_2x_low_res.py ${CHECKPOINT_PATH}
   ```

## Citation

<!-- You may remove this section if not applicable. -->

```latex
@inproceedings{
  xu2023nerfdet,
  title={NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection},
  author={Xu, Chenfeng and Wu, Bichen and Hou, Ji and Tsai, Sam and Li, Ruilong and Wang, Jialiang and Zhan, Wei and He, Zijian and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi},
  booktitle={ICCV},
  year={2023},
}

@inproceedings{
park2023time,
title={Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection},
author={Jinhyung Park and Chenfeng Xu and Shijia Yang and Kurt Keutzer and Kris M. Kitani and Masayoshi Tomizuka and Wei Zhan},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=H3HcEJA2Um}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet3d.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/models/detectors/fcos_mono3d.py) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tests/test_models/test_dense_heads/test_fcos_mono3d_head.py) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/fcos3d/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](/configs/textdet/dbnet/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
