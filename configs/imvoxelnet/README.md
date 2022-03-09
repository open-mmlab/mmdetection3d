# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

> [ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection](https://arxiv.org/abs/2106.01178)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we introduce the task of multi-view RGB-based 3D object detection as an end-to-end optimization problem. To address this problem, we propose ImVoxelNet, a novel fully convolutional method of 3D object detection based on posed monocular or multi-view RGB images. The number of monocular images in each multiview input can variate during training and inference; actually, this number might be unique for each multi-view input. ImVoxelNet successfully handles both indoor and outdoor scenes, which makes it general-purpose. Specifically, it achieves state-of-the-art results in car detection on KITTI (monocular) and nuScenes (multi-view) benchmarks among all methods that accept RGB images. Moreover, it surpasses existing RGB-based 3D object detection methods on the SUN RGB-D dataset. On ScanNet, ImVoxelNet sets a new benchmark for multi-view 3D object detection.

<div align=center>
<img src="https://user-images.githubusercontent.com/36950400/143871445-38a55168-b8cd-4520-8ed6-f5c8c8ea304a.png" width="800"/>
</div>

## Introduction

We implement a monocular 3D detector ImVoxelNet and provide its results and checkpoints on KITTI dataset.
Results for SUN RGB-D, ScanNet and nuScenes are currently available in ImVoxelNet authors
[repo](https://github.com/saic-vul/imvoxelnet) (based on mmdetection3d).

## Results and models

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: |:----: |
| [ResNet-50](./imvoxelnet_kitti-3d-car.py) | Car | 3x | | |17.4|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvoxelnet/imvoxelnet_kitti-3d-car_20210610_152323-b9abba85.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvoxelnet/imvoxelnet_kitti-3d-car_20210610_152323.log.json)|

## Citation

```latex
@article{rukhovich2021imvoxelnet,
  title={ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection},
  author={Danila Rukhovich, Anna Vorontsova, Anton Konushin},
  journal={arXiv preprint arXiv:2106.01178},
  year={2021}
}
```
