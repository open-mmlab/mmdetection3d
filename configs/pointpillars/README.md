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
|    [SECFPN](./hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py)|Car|cyclic 160e|5.4||77.1|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-77663cd6.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614.log.json)|
|    [SECFPN](./hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py)|3 Class|cyclic 160e|5.5||59.5|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421.log.json)|

### nuScenes

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](./hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||35.17|49.7|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725.log.json)|
|[FPN](./hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||40.0|53.3|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405.log.json)|

### Lyft

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](./hv_pointpillars_secfpn_sbn-all_4x8_2x_lyft-3d.py)|2x|||13.4|13.4||
|[FPN](./hv_pointpillars_fpn_sbn-all_4x8_2x_lyft-3d.py)|2x|||14.0|14.2||

### Waymo

|  Backbone | Load Interval | Class | Lr schd | Mem (GB) | Inf time (fps) | mAP@L1 | mAPH@L1 |  mAP@L2 | mAPH@L2 | Download |
| :-------: | :-----------: |:-----:| :------:| :------: | :------------: | :----: | :-----: | :-----: | :-----: | :------: |
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car.py)|5|Car|2x|||70.4|69.8|62.8|62.2||
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)|5|3 Class-Car|2x|||69.0|68.4|60.6|60.1||
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)|5|3 Class-Ped|2x|||67.7|51.1|59.9|45.1||
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)|5|3 Class-Cyc|2x|||55.9|52.8|53.8|50.8||

Here we provide 2 baselines for waymo dataset, which are adapted from the original pointpillars implementation. Specifically, we basically follow the implementation in the [paper](https://arxiv.org/pdf/1912.04838.pdf) in terms of the network architecture (having a
stride of 1 for the first convolutional block). Different settings of voxelization, data augmentation and hyper parameters make these baselines outperform those in the paper by about 7 mAP for car and 4 mAP for pedestrian with only a subset (1/5) of the whole dataset.
Note that using the complete dataset can boost the performance a lot, especially for the detection of pedestrian and cyclist, where more than 5 mAP improvement can be expected. A more complete benchmark with more models and methods is coming soon.
