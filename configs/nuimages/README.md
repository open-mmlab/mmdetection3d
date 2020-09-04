# PointPillars: Fast Encoders for Object Detection from Point Clouds

## Introduction

We implement [nuImages dataset](https://www.nuscenes.org/nuimages) and provide some baseline results on nuImages dataset.
We follow the class mapping in nuScenes dataset, which maps the original categories into 10 foreground categories.
The baseline results includes object detection and instance segmentation using ResNets and RegNets.
We will support panoptic segmentation models in the future.


## Results

### Instance Segmentation

We report Mask R-CNN and Cascade Mask R-CNN results on nuimages.

|Method |  Backbone| Lr schd | Mem (GB) | mask AP  | bbox AP  |Download |
| :---------: | :---------: | :-----: |:-----: | :------: | :------------: | :----: | :------: |
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_1x_nuim.py) ||||||
| Mask R-CNN| [R-101](./mask_rcnn_r101_fpn_1x_nuim.py) ||||||
| Mask R-CNN| [X-101_32x4d](./mask_rcnn_x101_32x4d_fpn_1x_nuim.py) ||||||
| Cascade Mask R-CNN| [R-50](./cascade_mask_rcnn_r50_fpn_1x_nuim.py) ||||||
| Cascade Mask R-CNN| [R-101](./cascade_mask_rcnn_r101_fpn_1x_nuim.py) ||||||
| Cascade Mask R-CNN| [X-101_32x4d](./cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py) ||||||
