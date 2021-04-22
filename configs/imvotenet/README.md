# ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes

## Introduction

<!-- [ALGORITHM] -->

We implement ImVoteNet and provide the result and checkpoints on SUNRGBD.

```
@inproceedings{qi2020imvotenet,
  title={Imvotenet: Boosting 3D object detection in point clouds with image votes},
  author={Qi, Charles R and Chen, Xinlei and Litany, Or and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4404--4413},
  year={2020}
}
```

## Results

### SUNRGBD-2D (Stage 1, image branch pre-train)

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py)     |   |2.1| ||62.70|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222.log.json)|

### SUNRGBD-3D (Stage 2)

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./imvotenet_stage2_16x8_sunrgbd-3d-10class.py)     |  3x    |9.4| |64.04||[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210323_184021-d44dcb66.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210323_184021.log.json)|
