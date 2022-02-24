# MVX-Net: Multimodal VoxelNet for 3D Object Detection

> [MVX-Net: Multimodal VoxelNet for 3D Object Detection](https://arxiv.org/abs/1904.01649)

<!-- [ALGORITHM] -->

## Abstract

Many recent works on 3D object detection have focused on designing neural network architectures that can consume point cloud data. While these approaches demonstrate encouraging performance, they are typically based on a single modality and are unable to leverage information from other modalities, such as a camera. Although a few approaches fuse data from different modalities, these methods either use a complicated pipeline to process the modalities sequentially, or perform late-fusion and are unable to learn interaction between different modalities at early stages. In this work, we present PointFusion and VoxelFusion: two simple yet effective early-fusion approaches to combine the RGB and point cloud modalities, by leveraging the recently introduced VoxelNet architecture. Evaluation on the KITTI dataset demonstrates significant improvements in performance over approaches which only use point cloud data. Furthermore, the proposed method provides results competitive with the state-of-the-art multimodal algorithms, achieving top-2 ranking in five of the six bird's eye view and 3D detection categories on the KITTI benchmark, by using a simple single stage network.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143880819-560675ca-e7e3-4d77-8808-ea661ff8e6e6.png" width="800"/>
</div>

## Introduction

We implement MVX-Net and provide its results and models on KITTI dataset.

## Results and models

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [SECFPN](./dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py)|3 Class|cosine 80e|6.7||63.0|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904.log.json)|

## Citation

```latex
@inproceedings{sindagi2019mvx,
  title={MVX-Net: Multimodal voxelnet for 3D object detection},
  author={Sindagi, Vishwanath A and Zhou, Yin and Tuzel, Oncel},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={7276--7282},
  year={2019},
  organization={IEEE}
}
```
