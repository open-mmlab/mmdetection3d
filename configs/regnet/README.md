# Designing Network Design Spaces

## Introduction

We implement RegNetX models in 3D detection systems and provide their first results with PointPillars on nuScenes dataset.

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
python -u tools/regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```
This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.


### Modify config

The users can modify the config's `depth` of backbone and corresponding keys in `arch` according to the configs in the [pycls model zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).
The parameter `in_channels` in FPN can be found in the Figure 15 & 16 of the paper (`wi` in the legend).
This directory already provides some configs with their performance, using RegNetX from 800MF to 12GF level.
For other pre-trained models or self-implemented regnet models, the users are responsible to check these parameters by themselves.

**Note**: Although Fig. 15 & 16 also provide `w0`, `wa`, `wm`, `group_w`, and `bot_mul` for `arch`, they are quantized thus inaccurate, using them sometimes produces different backbone that does not match the key in the pre-trained model.

## Results

### PointPillars

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [SECFPN](../)     |  2x    ||||
|[RegNetX-400MF-SECFPN](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d.py)|  2x    ||||
|    [FPN](../)     |  2x    ||||
|[RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py)|  2x    ||||
