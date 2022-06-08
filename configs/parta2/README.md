# From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network

> [From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network](https://arxiv.org/abs/1907.03670)

<!-- [ALGORITHM] -->

## Abstract

3D object detection from LiDAR point cloud is a challenging problem in 3D scene understanding and has many practical applications. In this paper, we extend our preliminary work PointRCNN to a novel and strong point-cloud-based 3D object detection framework, the part-aware and aggregation neural network (Part-A2 net). The whole framework consists of the part-aware stage and the part-aggregation stage. Firstly, the part-aware stage for the first time fully utilizes free-of-charge part supervisions derived from 3D ground-truth boxes to simultaneously predict high quality 3D proposals and accurate intra-object part locations. The predicted intra-object part locations within the same proposal are grouped by our new-designed RoI-aware point cloud pooling module, which results in an effective representation to encode the geometry-specific features of each 3D proposal. Then the part-aggregation stage learns to re-score the box and refine the box location by exploring the spatial relationship of the pooled intra-object part locations. Extensive experiments are conducted to demonstrate the performance improvements from each component of our proposed framework. Our Part-A2 net outperforms all existing 3D detection methods and achieves new state-of-the-art on KITTI 3D object detection dataset by utilizing only the LiDAR point cloud data.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143882774-6fc5f736-10d1-499a-8929-ca0768419049.png" width="800"/>
</div>

## Introduction

We implement Part-A^2 and provide its results and checkpoints on KITTI dataset.

## Results and models

### KITTI

|                            Backbone                            |  Class  |  Lr schd   | Mem (GB) | Inf time (fps) |  mAP  |                                                                                                                                                                                                   Download                                                                                                                                                                                                   |
| :------------------------------------------------------------: | :-----: | :--------: | :------: | :------------: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [SECFPN](./hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py) | 3 Class | cyclic 80e |   4.1    |                | 68.33 | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017-454a5344.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017.log.json) |
|  [SECFPN](./hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car.py)   |   Car   | cyclic 80e |   4.0    |                | 79.08 |       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017.log.json)       |

## Citation

```latex
@article{shi2020points,
  title={From points to parts: 3d object detection from point cloud with part-aware and part-aggregation network},
  author={Shi, Shaoshuai and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```
