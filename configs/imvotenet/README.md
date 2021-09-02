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

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP@0.25 |mAP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py)     |   |2.1| ||62.70|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618.log.json)|

### SUNRGBD-3D (Stage 2)

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP@0.25 |mAP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./imvotenet_stage2_16x8_sunrgbd-3d-10class.py)     |  3x    |9.4||64.55|38.65|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851-1bcd1b97.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851.log.json)|
