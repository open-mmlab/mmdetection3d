# CENet: Toward Concise and Efficient LiDAR Semantic Segmentation for Autonomous Driving

> [CENet: Toward Concise and Efficient LiDAR Semantic Segmentation for Autonomous Driving](https://arxiv.org/abs/2207.12691)

<!-- [ALGORITHM] -->

## Abstract

Accurate and fast scene understanding is one of the challenging task for autonomous driving, which requires to take full advantage of LiDAR point clouds for semantic segmentation. In this paper, we present a concise and efficient image-based semantic segmentation network, named CENet. In order to improve the descriptive power of learned features and reduce the computational as well as time complexity, our CENet integrates the convolution with larger kernel size instead of MLP, carefully-selected activation functions, and multiple auxiliary segmentation heads with corresponding loss functions into architecture. Quantitative and qualitative experiments conducted on publicly available benchmarks, SemanticKITTI and SemanticPOSS, demonstrate that our pipeline achieves much better mIoU and inference performance compared with state-of-the-art models. The code will be available at https://github.com/huixiancheng/CENet.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection3d/assets/55445986/2c268392-0e0c-4e93-bb9d-dc3417c56dad" width="800"/>
</div>

## Introduction

We implement CENet and provide the results and pretrained checkpoints on SemanticKITTI dataset.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDetection3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/CENet/configs/cenet-64x512_4xb4_semantickitti.py
```

For multi-gpu training, run:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/CENet/configs/cenet-64x512_4xb4_semantickitti.py
```

### Testing commands

In MMDetection3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/CENet/configs/cenet-64x512_4xb4_semantickitti.py ${CHECKPOINT_PATH}
```

## Results and models

### NuScenes

|                        Backbone                        | Input resolution | Mem (GB) | Inf time (fps) | mIoU  |         Download         |
| :----------------------------------------------------: | :--------------: | :------: | :------------: | :---: | :----------------------: |
| [CENet](./configs/cenet-64x512_4xb4_semantickitti.py)  |     64\*512      |          |      41.7      | 61.10 | [model](<>) \| [log](<>) |
| [CENet](./configs/cenet-64x1024_4xb4_semantickitti.py) |     64\*1024     |          |      26.8      | 62.20 | [model](<>) \| [log](<>) |
| [CENet](./configs/cenet-64x2048_4xb4_semantickitti.py) |     64\*2048     |          |      14.1      | 62.64 | [model](<>) \| [log](<>) |

**Note**

- We report point-based mIoU instead of range-view based mIoU
- The mIoU is the best results during inference after each epoch training, which is consistent with official code
- If your setting is different with our settings, we strongly suggest to enable `auto_scale_lr` to achieve comparable results.

## Citation

```latex
@inproceedings{cheng2022cenet,
  title={Cenet: Toward Concise and Efficient Lidar Semantic Segmentation for Autonomous Driving},
  author={Cheng, Hui--Xian and Han, Xian--Feng and Xiao, Guo--Qiang},
  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={01--06},
  year={2022},
  organization={IEEE}
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
