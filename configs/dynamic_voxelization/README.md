# Dynamic Voxelization

> [End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds](https://arxiv.org/abs/1910.06528)

<!-- [ALGORITHM] -->

## Abstract

Recent work on 3D object detection advocates point cloud voxelization in birds-eye view, where objects preserve their physical dimensions and are naturally separable. When represented in this view, however, point clouds are sparse and have highly variable point density, which may cause detectors difficulties in detecting distant or small objects (pedestrians, traffic signs, etc.). On the other hand, perspective view provides dense observations, which could allow more favorable feature encoding for such cases. In this paper, we aim to synergize the birds-eye view and the perspective view and propose a novel end-to-end multi-view fusion (MVF) algorithm, which can effectively learn to utilize the complementary information from both. Specifically, we introduce dynamic voxelization, which has four merits compared to existing voxelization methods, i) removing the need of pre-allocating a tensor with fixed size; ii) overcoming the information loss due to stochastic point/voxel dropout; iii) yielding deterministic voxel embeddings and more stable detection outcomes; iv) establishing the bi-directional relationship between points and voxels, which potentially lays a natural foundation for cross-view feature fusion. By employing dynamic voxelization, the proposed feature fusion architecture enables each point to learn to fuse context information from different views. MVF operates on points and can be naturally extended to other approaches using LiDAR point clouds. We evaluate our MVF model extensively on the newly released Waymo Open Dataset and on the KITTI dataset and demonstrate that it significantly improves detection accuracy over the comparable single-view PointPillars baseline.

<div align=center>
<img src="https://user-images.githubusercontent.com/30491025/143856017-98b77ecb-7c13-4164-9c1d-e3011a7645e6.png" width="600"/>
</div>

## Introduction

We implement Dynamic Voxelization proposed in  and provide its results and models on KITTI dataset.

## Results and models

### KITTI

|  Model   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: | :------: |
|[SECOND](./dv_second_secfpn_6x8_80e_kitti-3d-car.py)|Car    |cyclic 80e|5.5||78.83|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_second_secfpn_6x8_80e_kitti-3d-car/dv_second_secfpn_6x8_80e_kitti-3d-car_20200620_235228-ac2c1c0c.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_second_secfpn_6x8_80e_kitti-3d-car/dv_second_secfpn_6x8_80e_kitti-3d-car_20200620_235228.log.json)|
|[SECOND](./dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py)| 3 Class|cosine 80e|5.5||65.10|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010.log.json)|
|[PointPillars](./dv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py)| Car|cyclic 80e|4.7||77.76|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230844-ee7b75c9.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230844.log.json)|

## Citation

```latex
@article{zhou2019endtoend,
    title={End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds},
    author={Yin Zhou and Pei Sun and Yu Zhang and Dragomir Anguelov and Jiyang Gao and Tom Ouyang and James Guo and Jiquan Ngiam and Vijay Vasudevan},
    year={2019},
    eprint={1910.06528},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
