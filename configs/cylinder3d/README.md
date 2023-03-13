# Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation

> [Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation](https://arxiv.org/abs/2011.10033)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art methods for large-scale driving-scene LiDAR segmentation often project the point clouds to 2D space and then process them via 2D convolution. Although this corporation shows the competitiveness in the point cloud, it inevitably alters and abandons the 3D topology and geometric relations. A natural remedy is to utilize the3D voxelization and 3D convolution network. However, we found that in the outdoor point cloud, the improvement obtained in this way is quite limited. An important reason is the property of the outdoor point cloud, namely sparsity and varying density. Motivated by this investigation, we propose a new framework for the outdoor LiDAR segmentation, where cylindrical partition and asymmetrical 3D convolution networks are designed to explore the 3D geometric pat-tern while maintaining these inherent properties. Moreover, a point-wise refinement module is introduced to alleviate the interference of lossy voxel-based label encoding. We evaluate the proposed model on two large-scale datasets, i.e., SemanticKITTI and nuScenes. Our method achieves the 1st place in the leaderboard of SemanticKITTI and outperforms existing methods on nuScenes with a noticeable margin, about 4%. Furthermore, the proposed 3D framework also generalizes well to LiDAR panoptic segmentation and LiDAR 3D detection.

## Introduction

We implement Cylinder3D and provide the result and checkpoints on Semantickitti datasets.

## Results and models

### SemanticKITTI

|   Method   | Lr schd | Mem (GB) |   mIOU   |         Download         |
| :--------: | :-----: | :------: | :------: | :----------------------: |
| Cylinder3D |   3x    |   10.2   | 63.4±0.5 | [model](<>) \| [log](<>) |

## Citation

```latex
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}
```
