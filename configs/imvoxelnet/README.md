# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

## Introduction

<!-- [ALGORITHM] -->

We implement a monocular 3D detector ImVoxelNet and provide its results and checkpoints on KITTI dataset.
Results for SUN RGB-D, ScanNet and nuScenes are currently available in ImVoxelNet authors 
[repo](https://github.com/saic-vul/imvoxelnet) (based on mmdetection3d).

```
@article{rukhovich2021imvoxelnet,
  title={ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection},
  author={Danila Rukhovich, Anna Vorontsova, Anton Konushin},
  journal={arXiv preprint arXiv:2106.01178},
  year={2021}
}
```

## Results

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: |:----: |
| [ResNet-50](./imvoxelnet_kitti-3d-car.py) | Car | 3x | | |17.4|[model](https://disk.yandex.ru/d/PTz7-2H2Kby_Fw) &#124; [log](https://disk.yandex.ru/d/cj4JxnfRC6Yw4g)|
