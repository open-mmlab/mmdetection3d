# PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud

## Introduction

<!-- [ALGORITHM] -->

We implement PointRCNN and provide its results with checkpoints on KITTI dataset.

```
@InProceedings{Shi_2019_CVPR,
    author = {Shi, Shaoshuai and Wang, Xiaogang and Li, Hongsheng},
    title = {PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```

## Results

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: |:----: |
|    [PointNet++](./pointrcnn_2x8_kitti-3d-3classes.py) |3 Class|cyclic 80e|7.1||70.39||
