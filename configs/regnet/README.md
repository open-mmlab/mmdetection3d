# Designing Network Design Spaces

## Introduction

<!-- [BACKBONE] -->

We implement RegNetX models in 3D detection systems and provide their first results with PointPillars on nuScenes and Lyft dataset.

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md) and maintained in [mmcv](https://github.com/open-mmlab/mmcv).

```
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

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

## Results

### nuScenes

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||35.17|49.7|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725.log.json)|
|[RegNetX-400MF-SECFPN](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d.py)|  2x    |17.0||41.40|54.60|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20210827_013628-08b5cf1e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20210827_013628.log.json)|
|[FPN](../pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)|2x|17.1||40.0|53.3|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405.log.json)|
|[RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py)|2x|17.2||45.12|57.01|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20210827_095804-4239f111.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20210827_095804.log.json)|
|[RegNetX-1.6gF-FPN](./hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py)|2x|23.9||48.77|59.47|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20210827_013647-4ed9ea6b.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20210827_013647.log.json)|

### Lyft

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d.py)|2x|12.2||13.9|14.1|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807-2518e3de.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807.log.json)|
|[RegNetX-400MF-SECFPN](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_lyft-3d.py)| 2x |16.1||14.8|15.1|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d_20210828_124252-2624a744.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d_20210828_124252.log.json)|
|[RegNetX-400MF-SECFPN-Range100](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_range100_2x8_2x_lyft-3d.py)| 2x |24.5||15.9|16.0|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_range100_2x8_2x_lyft-3d_20210828_222452-4db050dc.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_range100_2x8_2x_lyft-3d_20210828_222452.log.json)|
|[FPN](../pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py)|2x|9.2||14.9|15.1|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818.log.json)|
|[RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_lyft-3d.py)|2x|13.1||16.2|16.4|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d_20210828_234451-79c31b63.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d_20210828_234451.log.json)|
|[RegNetX-400MF-FPN-Range100](./hv_pointpillars_regnet-400mf_fpn_sbn-all_range100_2x8_2x_lyft-3d.py)|2x|19.8||17.3|17.4|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_range100_2x8_2x_lyft-3d_20210828_115849-093f7ea6.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_range100_2x8_2x_lyft-3d_20210828_115849.log.json)|
