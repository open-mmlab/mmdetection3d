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
| :---------: | :-----: |:-----: | :------: | :------------: | :----: |:----: | :------: |
|    [SECFPN](./hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py) |3 Class|cyclic 80e|4.1||67.9||
|    [SECFPN](./hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car.py) |Car |cyclic 80e|4.0||79.16||
