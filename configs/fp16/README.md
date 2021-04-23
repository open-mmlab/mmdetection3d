# Mixed Precision Training

## Introduction

<!-- [OTHERS] -->

We implement mixed precision training and apply it to VoxelNets (e.g., SECOND and PointPillars).
The results are in the following tables.

**Note**: For mixed precision training, we currently do not support PointNet-based methods (e.g., VoteNet).
Mixed precision training for PointNet-based methods will be supported in the future release.

## Results

### SECOND on KITTI dataset

|  Backbone   |Class| Lr schd | FP32 Mem (GB) | FP16 Mem (GB) | FP32 mAP | FP16 mAP |Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: | :------: |
|    [SECFPN](./hv_second_secfpn_fp16_6x8_80e_kitti-3d-car.py)| Car |cyclic 80e|5.4|2.9|79.07|78.72|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car_20200924_211301-1f5ad833.pth)&#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car_20200924_211301.log.json)|
|    [SECFPN](./hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py)| 3 Class |cyclic 80e|5.4|2.9|64.41|67.4|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059.log.json)|

### PointPillars on nuScenes dataset

|  Backbone   | Lr schd | FP32 Mem (GB) | FP16 Mem (GB) | FP32 mAP | FP32 NDS| FP16 mAP | FP16 NDS| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :----: |:----: | :------: |
|[SECFPN](./hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d.py)|2x|16.4|8.37|35.17|49.7|35.19|50.27|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626.log.json)|
|[FPN](./hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py)|2x|16.4|8.40|40.0|53.3|39.26|53.26|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719.log.json)|

**Note**:
1. With mixed precision training, we can train PointPillars with nuScenes dataset on 8 Titan XP GPUS with batch size of 2.
This will cause OOM error without mixed precision training.
2. The loss scale for PointPillars on nuScenes dataset is specifically tuned to avoid the loss to be Nan. We find 32 is more stable than 512, though loss scale 32 still cause Nan sometimes.
