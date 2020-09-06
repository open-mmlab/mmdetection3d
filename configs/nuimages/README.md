# PointPillars: Fast Encoders for Object Detection from Point Clouds

## Introduction

We implement [nuImages dataset](https://www.nuscenes.org/nuimages) and provide some baseline results on nuImages dataset.
We follow the class mapping in nuScenes dataset, which maps the original categories into 10 foreground categories.
The baseline results includes object detection and instance segmentation using ResNets and RegNets.
We will support panoptic segmentation models in the future.


## Results

### Instance Segmentation

We report Mask R-CNN and Cascade Mask R-CNN results on nuimages.

|Method |  |Backbone| Lr schd | Mem (GB) | mask AP  | bbox AP  |Download |
| :---------: |:---------: | :---------: | :-----: |:-----: | :------: | :------------: | :----: | :------: |
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_1x_nuim.py) |IN|1x||48.0 |38.5|||
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_coco-2x_1x_nuim.py) |IN+COCO-2x|1x||49.5|40.0||
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_1x_nuim.py) |IN|1x|||||
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim.py) |IN+COCO-3x|1x||49.6|40.3||
| Mask R-CNN| [R-101](./mask_rcnn_r101_fpn_1x_nuim.py) |IN|1x|||||
| Mask R-CNN| [X-101_32x4d](./mask_rcnn_x101_32x4d_fpn_1x_nuim.py) |IN|1x|||||
| Cascade Mask R-CNN| [R-50](./cascade_mask_rcnn_r50_fpn_1x_nuim.py) |IN|1x|||||
| Cascade Mask R-CNN| [R-101](./cascade_mask_rcnn_r101_fpn_1x_nuim.py) |IN|1x|||||
| Cascade Mask R-CNN| [X-101_32x4d](./cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py) |IN|1x|||||

**Note**:
1. `IN` means only using ImageNet pre-trained backbone. `IN+COCO-Nx` means the backbone is first pre-trained on ImageNet, and then the detector is pre-trained on COCO train2017 dataset by `Nx` schedules.
2. All the training hyper-parameters follows the standard 1x schedules on COCO dataset except that the images are resized from
1280 x 720 to 1920 x 1080 (relative ratio 0.8 to 1.2) since the images are in size 1600 x 900.
