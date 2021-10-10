# SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation

## Introduction

<!-- [ALGORITHM] -->

We implement SMOKE and provide the results and checkpoints on KITTI dataset.

```
@article{liu2020SMOKE,
  title={{SMOKE}: Single-Stage Monocular 3D Object Detection via Keypoint Estimation},
  author={Zechen Liu and Zizhang Wu and Roland T\'oth},
  journal={arXiv preprint arXiv:2002.10111},
  year={2020}
}
```

## Results

### KITTI

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: | :------: | :------------: | :----: | :------: |
|[DLA34](./smoke_dla34_pytorch_dlaneck_gn-head_kitti_mono3d.py)|6x|9.64||13.85|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553.log.json)

Note: mAP represents Car moderate 3D strict AP11 results.

Detailed performance on KITTI 3D detection (3D/BEV) is as follows:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 16.92 / 22.97 | 13.85 / 18.32  | 11.90 / 15.88|
| Pedestrian  | 11.13  / 12.61| 11.10 / 11.32  | 10.67 / 11.14|
| Cyclist     | 0.99  / 1.47  | 0.54 / 0.65    | 0.55 / 0.67  |
