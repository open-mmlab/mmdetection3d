# Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution

> [Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution ](https://arxiv.org/abs/2007.16100)

<!-- [ALGORITHM] -->

## Abstract

Self-driving cars need to understand 3D scenes efficiently and accurately in order to drive safely. Given the limited hardware resources, existing 3D perception models are not able to recognize small instances (e.g., pedestrians, cyclists) very well due to the low-resolution voxelization and aggressive downsampling. To this end, we propose Sparse Point-Voxel Convolution (SPVConv), a lightweight 3D module that equips the vanilla Sparse Convolution with the high-resolution point-based branch. With negligible overhead, this point-based branch is able to preserve the fine details even from large outdoor scenes. To explore the spectrum of efficient 3D models, we first define a flexible architecture design space based on SPVConv, and we then present 3D Neural Architecture Search (3D-NAS) to search the optimal network architecture over this diverse design space efficiently and effectively. Experimental results validate that the resulting SPVNAS model is fast and accurate: it outperforms the state-of-the-art MinkowskiNet by 3.3%, ranking 1st on the competitive SemanticKITTI leaderboard. It also achieves 8x computation reduction and 3x measured speedup over MinkowskiNet with higher accuracy. Finally, we transfer our method to 3D object detection, and it achieves consistent improvements over the one-stage detection baseline on KITTI.

<div align=center>
<img src="https://user-images.githubusercontent.com/72679458/226509154-80c27d8e-c138-426a-b92e-72846997b5b3.png" width="800"/>
</div>

## Introduction

We implement SPVCNN with [TorchSparse](https://github.com/mit-han-lab/torchsparse) backend and provide the result and checkpoints on SemanticKITTI datasets.

## Results and models

### SemanticKITTI

|                                 Method                                  | Lr schd | Laser-Polar Mix | Mem (GB) | mIoU |                                                                                                                                                                    Download                                                                                                                                                                     |
| :---------------------------------------------------------------------: | :-----: | :-------------: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        [SPVCNN-W16](./spvcnn_w16_8xb2-amp-15e_semantickitti.py)         |   15e   |        ✗        |   3.9    | 61.8 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w16_8xb2-15e_semantickitti/spvcnn_w16_8xb2-15e_semantickitti_20230321_011645-a2734d85.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w16_8xb2-15e_semantickitti/spvcnn_w16_8xb2-15e_semantickitti_20230321_011645.log) |
|        [SPVCNN-W20](./spvcnn_w20_8xb2-amp-15e_semantickitti.py)         |   15e   |        ✗        |   4.2    | 62.6 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w20_8xb2-15e_semantickitti/spvcnn_w20_8xb2-15e_semantickitti_20230321_011649-519e7eff.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w20_8xb2-15e_semantickitti/spvcnn_w20_8xb2-15e_semantickitti_20230321_011649.log) |
|        [SPVCNN-W32](./spvcnn_w32_8xb2-amp-15e_semantickitti.py)         |   15e   |        ✗        |   5.4    | 64.3 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w32_8xb2-15e_semantickitti/spvcnn_w32_8xb2-15e_semantickitti_20230308_113324-f7c0c5b4.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w32_8xb2-15e_semantickitti/spvcnn_w32_8xb2-15e_semantickitti_20230308_113324.log) |
| [SPVCNN-W32](./spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti.py) |   3x    |        ✔        |   7.2    | 68.7 |                [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_125908-d68a68b7.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/spvcnn/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_125908.log)                |

**Note:** We follow the implementation in SPVNAS original [repo](https://github.com/mit-han-lab/spvnas) and W16\\W20\\W32 indicates different number of channels.

**Note:** Due to TorchSparse backend, the model performance is unstable with TorchSparse backend and may fluctuate by about 1.5 mIoU for different random seeds.

## Citation

```latex
@inproceedings{tang2020searching,
  title={Searching efficient 3d architectures with sparse point-voxel convolution},
  author={Tang, Haotian and Liu, Zhijian and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XXVIII},
  pages={685--702},
  year={2020},
  organization={Springer}
}
```
