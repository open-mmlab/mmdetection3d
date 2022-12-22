# Dummy PETR

This is an README for `dummy_PETR`.

## Description

Author: @SekiroRong.
This is an implementation of *PETR*.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/dummy_PETR/config/petr/petr_vovnet_gridmask_p4_800x320.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/dummy_PETR/config/petr/petr_vovnet_gridmask_p4_800x320.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmdetection3d/edit/dev-1.x/configs/fcos3d/README.md)
 You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

This Result is trained by petr_vovnet_gridmask_p4_800x320.py and use [weights](https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view?usp=sharing) as pretrain weight.

|                                                Backbone                                                | Lr schd | Mem (GB) | Inf time (fps) | mAP  | NDS  |         Download         |
| :----------------------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :--: | :--: | :----------------------: |
| [petr_vovnet_gridmask_p4_800x320](projects/dummy_PETR/configs/petr/petr_vovnet_gridmask_p4_800x320.py) |   1x    |   7.62   |      18.7      | 38.3 | 43.5 | [model](<>) \| [log](<>) |

```
mAP: 0.3829
mATE: 0.7376
mASE: 0.2702
mAOE: 0.4803
mAVE: 0.8703
mAAE: 0.2040
NDS: 0.4352
Eval time: 117.6s

Per-class results:
Object Class	  AP	  ATE	  ASE	  AOE	  AVE	  AAE
car	  0.574	  0.519	  0.150	  0.087	  0.866	  0.206
truck	  0.349	  0.774	  0.213	  0.117	  0.855	  0.221
bus	  0.424	  0.782	  0.204	  0.123	  1.904	  0.319
trailer 0.219	  1.035	  0.231	  0.609	  0.830	  0.149
construction_vehicle	  0.084	  1.058	  0.485	  1.248	  0.172	  0.361
pedestrian	  0.452	  0.682	  0.293	  0.645	  0.529	  0.231
motorcycle	  0.378	  0.671	  0.250	  0.567	  1.334	  0.130
bicycle	      0.347	  0.640	  0.264	  0.788	  0.473	  0.016
traffic_cone	  0.538	  0.553	  0.325	  nan	  nan	  nan
barrier	      0.463	  0.663	  0.287	  0.138	  nan	  nan
```
