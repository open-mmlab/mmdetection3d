# ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes

> [ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes](https://arxiv.org/abs/2001.10692)

<!-- [ALGORITHM] -->

## Abstract

3D object detection has seen quick progress thanks to advances in deep learning on point clouds. A few recent works have even shown state-of-the-art performance with just point clouds input (e.g. VOTENET). However, point cloud data have inherent limitations. They are sparse, lack color information and often suffer from sensor noise. Images, on the other hand, have high resolution and rich texture. Thus they can complement the 3D geometry provided by point clouds. Yet how to effectively use image information to assist point cloud based detection is still an open question. In this work, we build on top of VOTENET and propose a 3D detection architecture called IMVOTENET specialized for RGB-D scenes. IMVOTENET is based on fusing 2D votes in images and 3D votes in point clouds. Compared to prior work on multi-modal detection, we explicitly extract both geometric and semantic features from the 2D images. We leverage camera parameters to lift these features to 3D. To improve the synergy of 2D-3D feature fusion, we also propose a multi-tower training scheme. We validate our model on the challenging SUN RGB-D dataset, advancing state-of-the-art results by 5.7 mAP. We also provide rich ablation studies to analyze the contribution of each design choice.

<div align=center>
<img src="https://user-images.githubusercontent.com/36950400/143869878-a2ae7f43-55c3-4b95-af09-8f97dfd975f4.png" width="800"/>
</div>

## Introduction

We implement ImVoteNet and provide the result and checkpoints on SUNRGBD.

## Results and models

### SUNRGBD-2D (Stage 1, image branch pre-train)

|                                Backbone                                 | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 | AP@0.5 |                                                                                                                                                                                                              Download                                                                                                                                                                                                              |
| :---------------------------------------------------------------------: | :-----: | :------: | :------------: | :-----: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PointNet++](./imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py) |         |   2.1    |                |         | 62.70  | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618.json) |

### SUNRGBD-3D (Stage 2)

|                          Backbone                           | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 | AP@0.5 |                                                                                                                                                                                        Download                                                                                                                                                                                        |
| :---------------------------------------------------------: | :-----: | :------: | :------------: | :-----: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PointNet++](./imvotenet_stage2_16x8_sunrgbd-3d-10class.py) |   3x    |   9.4    |                |  64.55  |        | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851-1bcd1b97.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851.log.json) |

## Citation

```latex
@inproceedings{qi2020imvotenet,
  title={Imvotenet: Boosting 3D object detection in point clouds with image votes},
  author={Qi, Charles R and Chen, Xinlei and Litany, Or and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4404--4413},
  year={2020}
}
```
