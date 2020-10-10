# Mixed Precision Training

## Introduction

We implement mixed precision training and apply it to VoxelNets (e.g., SECOND and PointPillars).
The results are in the following tables.

**Note**: For mixed precision training, we currently do not support PointNet-based methods (e.g., VoteNet).
Mixed precision training for PointNet-based methods will be supported in the future release.

## Results

### SECOND on KITTI dataset
|  Backbone   |Class| Lr schd | FP32 Mem (GB) | FP16 Mem (GB) | FP32 mAP | FP16 mAP |Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: | :------: |
|    [SECFPN](./hv_second_secfpn_fp16_6x8_80e_kitti-3d-car.py)| Car |cyclic 80e|5.4|2.9|79.07|78.72||
|    [SECFPN](./hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py)| 3 Class |cyclic 80e|5.4|2.9|64.41|67.4||

### PointPillars on nuScenes dataset


**Note**: With mixed precision training, we can train PointPillars with RegNet-400mf on 8 Titan XP GPUS with batch size of 2.
This will cause OOM error without mixed precision training.
