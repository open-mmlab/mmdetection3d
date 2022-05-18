# Designing Network Design Spaces

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

<!-- [BACKBONE] -->

## Abstract

In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/144025148-b73002cb-3c82-42e4-8da4-65df97aead9c.png" width="800"/>
</div>

## Introduction

We implement RegNetX models in 3D detection systems and provide their first results with PointPillars on nuScenes and Lyft dataset.

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md) and maintained in [mmcv](https://github.com/open-mmlab/mmcv).

## Usage

To use a regnet model, there are two steps to do:

1. Convert the model to ResNet-style supported by MMDetection
2. Modify backbone and neck in config accordingly

### Convert model

We already prepare models of FLOPs from 800M to 12G in our model zoo.

For more general usage, we also provide script `regnet2mmdet.py` in the tools directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
ResNet-style checkpoints used in MMDetection.

```bash
python -u tools/model_converters/regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

### Modify config

The users can modify the config's `depth` of backbone and corresponding keys in `arch` according to the configs in the [pycls model zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).
The parameter `in_channels` in FPN can be found in the Figure 15 & 16 of the paper (`wi` in the legend).
This directory already provides some configs with their performance, using RegNetX from 800MF to 12GF level.
For other pre-trained models or self-implemented regnet models, the users are responsible to check these parameters by themselves.

**Note**: Although Fig. 15 & 16 also provide `w0`, `wa`, `wm`, `group_w`, and `bot_mul` for `arch`, they are quantized thus inaccurate, using them sometimes produces different backbone that does not match the key in the pre-trained model.

## Results and models

### nuScenes

|                                        Backbone                                        | Lr schd | Mem (GB) | Inf time (fps) |  mAP  | NDS  |                                                                                                                                                                                                                       Download                                                                                                                                                                                                                       |
| :------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :---: | :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)        |   2x    |   16.4   |                | 35.17 | 49.7 |                     [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725.log.json)                     |
| [RegNetX-400MF-SECFPN](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d.py) |   2x    |   16.4   |                | 41.2  | 55.2 | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334-53044f32.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334.log.json) |
|          [FPN](../pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)           |   2x    |   17.1   |                | 40.0  | 53.3 |                           [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405.log.json)                           |
|    [RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py)    |   2x    |   17.3   |                | 44.8  | 56.4 |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20200620_230239-c694dce7.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20200620_230239.log.json)       |
|    [RegNetX-1.6gF-FPN](./hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py)    |   2x    |   24.0   |                | 48.2  | 59.3 |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311.log.json)       |

### Lyft

|                                        Backbone                                         | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score |                                                                                                                                                                                                                         Download                                                                                                                                                                                                                         |
| :-------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :-----------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d.py)        |   2x    |   12.2   |                |     13.9      |     14.1     |                     [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807-2518e3de.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807.log.json)                     |
| [RegNetX-400MF-SECFPN](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_lyft-3d.py) |   2x    |   15.9   |                |     14.9      |     15.1     | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d_20210524_092151-42513826.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d_20210524_092151.log.json) |
|          [FPN](../pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py)           |   2x    |   9.2    |                |     14.9      |     15.1     |                           [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818.log.json)                           |
|    [RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_lyft-3d.py)    |   2x    |   13.0   |                |     16.0      |     16.1     |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d_20210521_115618-823dcf18.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d_20210521_115618.log.json)       |

## Citation

```latex
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
