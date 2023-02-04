# PETR

This is an README for `PETR`.

## Description

Author: @SekiroRong.
This is an implementation of *PETR*.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/PETR/config/petr/petr_vovnet_gridmask_p4_800x320.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/PETR/config/petr/petr_vovnet_gridmask_p4_800x320.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmdetection3d/edit/dev-1.x/configs/fcos3d/README.md)
 You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

This Result is trained by petr_vovnet_gridmask_p4_800x320.py and use [weights](https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view?usp=sharing) as pretrain weight.

|                                             Backbone                                             | Lr schd | Mem (GB) | Inf time (fps) | mAP  | NDS  |         Download         |
| :----------------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :--: | :--: | :----------------------: |
| [petr_vovnet_gridmask_p4_800x320](projects/PETR/configs/petr/petr_vovnet_gridmask_p4_800x320.py) |   1x    |   7.62   |      18.7      | 38.3 | 43.5 | [model](<>) \| [log](<>) |

```
mAP: 0.3830
mATE: 0.7547
mASE: 0.2683
mAOE: 0.4948
mAVE: 0.8331
mAAE: 0.2056
NDS: 0.4358
Eval time: 118.7s

Per-class results:
Object Class	  AP	  ATE	  ASE	  AOE	  AVE	  AAE
car	  0.567	  0.538	  0.151	  0.086	  0.873	  0.212
truck	  0.341	  0.785	  0.213	  0.113	  0.821	  0.234
bus	  0.426	  0.766	  0.201	  0.128	  1.813	  0.343
trailer 0.216	  1.116	  0.227	  0.649	  0.640	  0.122
construction_vehicle	  0.093	  1.118	  0.483	  1.292	  0.217	  0.330
pedestrian	  0.453	  0.685	  0.293	  0.644	  0.535	  0.238
motorcycle	  0.374	  0.700	  0.253	  0.624	  1.291	  0.154
bicycle	      0.345	  0.622	  0.262	  0.775	  0.475	  0.011
traffic_cone	  0.539	  0.557	  0.319	  nan	  nan	  nan
barrier	      0.476	  0.661	  0.279	  0.142	  nan	  nan
```
