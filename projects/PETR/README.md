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
python tools/train.py projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmdetection3d/edit/dev-1.x/configs/fcos3d/README.md)
 You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

This Result is trained by petr_vovnet_gridmask_p4_800x320.py and use [weights](https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view?usp=sharing) as pretrain weight.

|                                   Backbone                                    | Lr schd | Mem (GB) | Inf time (fps) | mAP  | NDS  |                                                                                                      Download                                                                                                       |
| :---------------------------------------------------------------------------: | :-----: | :------: | :------------: | :--: | :--: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [petr_vovnet_gridmask_p4_800x320](configs/petr_vovnet_gridmask_p4_800x320.py) |   1x    |   7.62   |      18.7      | 38.3 | 43.5 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/petr_vovnet_gridmask_p4_800x320-e2191752.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/20221222_232156.log) |
