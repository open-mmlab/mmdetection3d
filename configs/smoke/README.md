# SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation

## Introduction

<!-- [ALGORITHM] -->

We implement SMOKE and provide the results and checkpoints on KITTI dataset.

```
@inproceedings{liu2020smoke,
  title={Smoke: Single-stage monocular 3d object detection via keypoint estimation},
  author={Liu, Zechen and Wu, Zizhang and T{\'o}th, Roland},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={996--997},
  year={2020}
}
```

## Results

### KITTI

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: | :------: | :------------: | :----: | :------: |
|[DLA34](./smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py)|6x|9.64||13.85|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553.log.json)

Note: mAP represents Car moderate 3D strict AP11 results.

Detailed performance on KITTI 3D detection (3D/BEV) is as follows, evaluated by AP11 metric:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 16.92 / 22.97 | 13.85 / 18.32  | 11.90 / 15.88|
| Pedestrian  | 11.13 / 12.61| 11.10 / 11.32  | 10.67 / 11.14|
| Cyclist     | 0.99  / 1.47  | 0.54 / 0.65    | 0.55 / 0.67  |
