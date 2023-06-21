# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

> [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://arxiv.org/abs/2110.06922)

<!-- [ALGORITHM] -->

## Abstract

We introduce a framework for multi-camera 3D object detection. In
contrast to existing works, which estimate 3D bounding boxes directly from
monocular images or use depth prediction networks to generate input for 3D object
detection from 2D information, our method manipulates predictions directly
in 3D space. Our architecture extracts 2D features from multiple camera images
and then uses a sparse set of 3D object queries to index into these 2D features,
linking 3D positions to multi-view images using camera transformation matrices.
Finally, our model makes a bounding box prediction per object query, using a
set-to-set loss to measure the discrepancy between the ground-truth and the prediction.
This top-down approach outperforms its bottom-up counterpart in which
object bounding box prediction follows per-pixel depth estimation, since it does
not suffer from the compounding error introduced by a depth prediction model.
Moreover, our method does not require post-processing such as non-maximum
suppression, dramatically improving inference speed. We achieve state-of-the-art
performance on the nuScenes autonomous driving benchmark.

<div align=center>
<img src="https://user-images.githubusercontent.com/67246790/209751755-3d0f0ad5-6a39-4d14-a1c7-346b5c228a1b.png" width="800"/>
</div>

## Introduction

This directory contains the implementations of DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.
We have updated DETR3D to be compatible with latest mmdet3d-dev1.x. The codebase and config files have all changed to adapt to the new mmdet3d version. All previous pretrained models are verified with the result listed below. However, newly trained models are yet to be uploaded.

## Environment Setup

We require the version of mmdet \<= V3.0.0rc5. The mmdet later than V3.0.0rc5 has refactored DETR-series and its config file, but our configs and code are yet to be updated.

## Train

1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/

2. For example, to train DETR3D on 8 GPUs, please use

```bash
bash tools/dist_train.sh projects/DETR3D/configs/detr3d_res101_gridmask.py 8 --cfg-options load_from=pretrained/fcos3d.pth
```

## Evaluation using pretrained models

1. Download the newly trained weights accordingly.

   |                                                Backbone                                                 | mAP  | NDS  |                                                                                                                 Download                                                                                                                 |
   | :-----------------------------------------------------------------------------------------------------: | :--: | :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
   |          [DETR3D, ResNet101 w/ DCN, evaluation on val set](./configs/detr3d_r101_gridmask.py)           | 35.5 | 42.8 |                 [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/detr3d/detr3d_r101_gridmask.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/detr3d/detr3d_r101_gridmask.log)                 |
   |             [above, + CBGS, evaluation on val set](./configs/detr3d_r101_gridmask_cbgs.py)              | 35.2 | 42.7 |            [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/detr3d/detr3d_r101_gridmask_cbgs.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/detr3d/detr3d_r101_gridmask_cbgs.log)            |
   | [DETR3D, VoVNet on trainval, evaluation on test set](./configs/detr3d_vovnet_gridmask_trainval_cbgs.py) | 41.4 | 48.1 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/detr3d/detr3d_vovnet_gridmask_trainval_cbgs.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/detr3d/detr3d_vovnet_gridmask_trainval_cbgs.log) |

2. Testing

   To test, use:

   ```bash
    bash tools/dist_test.sh projects/DETR3D/configs/detr3d_res101_gridmask.py ${CHECKPOINT_PATH} 8
   ```

## Converting old models (Optional)

For old models please refer to [Object DGCNN & DETR3D](https://github.com/WangYueFt/detr3d)

From v0.17.3 to v1.0.0, mmdet3d has changed its bbox representation. Given that Box(x,y,z,θ), we have x_new = y_old, y_new = x_old, θ_new = -θ_old - π/2.

Old models are trained on v0.17.3. Our regression branch outputs (cx,cy,w,l,cz,h,sin(θ),cos(θ),vx,vy). For a previous model which outputs y=\[y0,y1,y2,y3,y4,y5,y6,y7,y8,y9\], we get y_new = \[...,y3,y2,...,-y7,-y6, ...\]. So we should change the final Linear layer's weight accordingly.

To convert the old weights, please use

```bash
python projects/DETR3D/detr3d/old_detr3d_converter.py ${CHECKPOINT_DIR}/detr3d_resnet101.pth ${CHECKPOINT_DIR}/detr3d_r101_v1.0.0.pth --code_size 10
```

## Citation

If you find this repo useful for your research, please consider citing the papers

```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
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

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

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
