# PointPillars: Fast Encoders for Object Detection from Point Clouds

## Introduction

We implement PointPillars and provide the results and checkpoints on KITTI and nuScenes datasets.

```
@inproceedings{lang2019pointpillars,
  title={Pointpillars: Fast encoders for object detection from point clouds},
  author={Lang, Alex H and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12697--12705},
  year={2019}
}

```

## Results

### KITTI

|  Backbone|Class   | Lr schd | Mem (GB) | Inf time (fps) | AP  |Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: | :------: |
|    [SECFPN](./hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py)|Car|cyclic 160e|5.4||77.1||
|    [SECFPN](./hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py)|3 Class|cyclic 160e|5.5||59.5|

### nuScenes

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](./hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||35.17|49.7||
|[FPN](./hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||40.0|53.3||
