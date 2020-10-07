# MVX-Net: Multimodal VoxelNet for 3D Object Detection

## Introduction

We implement MVX-Net and provide its results and models on KITTI dataset.
```
@inproceedings{sindagi2019mvx,
  title={MVX-Net: Multimodal voxelnet for 3D object detection},
  author={Sindagi, Vishwanath A and Zhou, Yin and Tuzel, Oncel},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={7276--7282},
  year={2019},
  organization={IEEE}
}

```

## Results

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [SECFPN](./dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py)|3 Class|cosine 80e|6.7||63.0|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904.log.json)|
