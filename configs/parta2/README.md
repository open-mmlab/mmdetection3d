# From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network

## Introduction

We implement Part-A^2 and provide its results and checkpoints on KITTI dataset.

```
@article{shi2020points,
  title={From points to parts: 3d object detection from point cloud with part-aware and part-aggregation network},
  author={Shi, Shaoshuai and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}

```
## Results

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: |:----: |
|    [SECFPN](./hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py) |3 Class|cyclic 80e|4.1||67.9|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724.log.json)|
|    [SECFPN](./hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car.py) |Car |cyclic 80e|4.0||79.16|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20200620_230755-f2a38b9a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20200620_230755.log.json)|
