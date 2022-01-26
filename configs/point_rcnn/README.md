# PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud

> [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we propose PointRCNN for 3D object detection from raw point cloud. The whole framework is composed of two stages: stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in the canonical coordinates to obtain the final detection results. Instead of generating proposals from RGB image or projecting point cloud to bird's view or voxels as previous methods do, our stage-1 sub-network directly generates a small number of high-quality 3D proposals from point cloud in a bottom-up manner via segmenting the point cloud of the whole scene into foreground points and background. The stage-2 sub-network transforms the pooled points of each proposal to canonical coordinates to learn better local spatial features, which is combined with global semantic features of each point learned in stage-1 for accurate box refinement and confidence prediction. Extensive experiments on the 3D detection benchmark of KITTI dataset show that our proposed architecture outperforms state-of-the-art methods with remarkable margins by using only point cloud as input.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/144959105-271038a2-4ae1-4cdb-b6a8-68c14daf83b0.png" width="800"/>
</div>

## Introduction

We implement PointRCNN and provide the result with checkpoints on KITTI dataset.

## Results and models

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: |:----: |
|    [PointNet++](./point_rcnn_2x8_kitti-3d-3classes.py) |3 Class|cyclic 40e|4.6||70.83|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/point_rcnn/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/point_rcnn/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.log.json)|

Note: mAP represents AP11 results on 3 Class under the moderate setting.

Detailed performance on KITTI 3D detection (3D) is as follows, evaluated by AP11 metric:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 89.13 | 78.72 | 78.24 |
| Pedestrian  | 65.81 | 59.57 | 52.75 |
| Cyclist     | 93.51 | 74.19 | 70.73 |

## Citation

```latex
@inproceedings{Shi_2019_CVPR,
    title = {PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud},
    author = {Shi, Shaoshuai and Wang, Xiaogang and Li, Hongsheng},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```
