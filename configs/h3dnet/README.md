# H3DNet: 3D Object Detection Using Hybrid Geometric Primitives

## Introduction

<!-- [ALGORITHM] -->

We implement H3DNet and provide the result and checkpoints on ScanNet datasets.

```
@inproceedings{zhang2020h3dnet,
    author = {Zhang, Zaiwei and Sun, Bo and Yang, Haitao and Huang, Qixing},
    title = {H3DNet: 3D Object Detection Using Hybrid Geometric Primitives},
    booktitle = {Proceedings of the European Conference on Computer Vision},
    year = {2020}
}
```

## Results

### ScanNet

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP@0.25 |mAP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [MultiBackbone](./h3dnet_3x8_scannet-3d-18class.py)     |  3x    |7.9||66.07|47.68|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/h3dnet/h3dnet_3x8_scannet-3d-18class/h3dnet_3x8_scannet-3d-18class_20210824_003149-414bd304.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/h3dnet/h3dnet_3x8_scannet-3d-18class/h3dnet_3x8_scannet-3d-18class_20210824_003149.log.json) |
