# Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation

> [Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation](https://arxiv.org/abs/2011.10033)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art methods for large-scale driving-scene LiDAR segmentation often project the point clouds to 2D space and then process them via 2D convolution. Although this corporation shows the competitiveness in the point cloud, it inevitably alters and abandons the 3D topology and geometric relations. A natural remedy is to utilize the3D voxelization and 3D convolution network. However, we found that in the outdoor point cloud, the improvement obtained in this way is quite limited. An important reason is the property of the outdoor point cloud, namely sparsity and varying density. Motivated by this investigation, we propose a new framework for the outdoor LiDAR segmentation, where cylindrical partition and asymmetrical 3D convolution networks are designed to explore the 3D geometric pat-tern while maintaining these inherent properties. Moreover, a point-wise refinement module is introduced to alleviate the interference of lossy voxel-based label encoding. We evaluate the proposed model on two large-scale datasets, i.e., SemanticKITTI and nuScenes. Our method achieves the 1st place in the leaderboard of SemanticKITTI and outperforms existing methods on nuScenes with a noticeable margin, about 4%. Furthermore, the proposed 3D framework also generalizes well to LiDAR panoptic segmentation and LiDAR 3D detection.

![overview](https://user-images.githubusercontent.com/45515569/228523861-2923082c-37d9-4d4f-aa59-746a8d9284c2.png)

## Introduction

We implement Cylinder3D and provide the result and checkpoints on Semantickitti datasets.

## Results and models

### SemanticKITTI

|                               Method                                | Lr schd | Laser-Polar Mix | Mem (GB) |   mIoU   |                                                                                                                                                                       Download                                                                                                                                                                       |
| :-----------------------------------------------------------------: | :-----: | :-------------: | :------: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Cylinder3D](./cylinder3d_8xb2-laser-polar-mix-3x_semantickitti.py) |   3x    |        ✗        |   10.2   | 63.1±0.5 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/cylinder3d/cylinder3d_4xb4_3x_semantickitti/cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/cylinder3d/cylinder3d_4xb4_3x_semantickitti/cylinder3d_4xb4_3x_semantickitti_20230318_191107.json) |
| [Cylinder3D](./cylinder3d_8xb2-laser-polar-mix-3x_semantickitti.py) |   3x    |        ✔        |   12.8   |   67.0   |              [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/cylinder3d/cylinder3d_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_144950-372cdf69.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/cylinder3d/cylinder3d_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_144950.log)               |

Note: We reproduce the performance comparable with its [official repo](https://github.com/xinge008/Cylinder3D). It's slightly lower than the performance (65.9 mIOU) reported in the paper due to the lack of point-wise refinement and shorter training time.

## Citation

```latex
@inproceedings{zhu2021cylindrical,
  title={Cylindrical and asymmetrical 3d convolution networks for lidar segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={9939--9948},
  year={2021}
}
```
