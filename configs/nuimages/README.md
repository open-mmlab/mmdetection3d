# NuImages Results

## Introduction

We support and provide some baseline results on [nuImages dataset](https://www.nuscenes.org/nuimages).
We follow the class mapping in nuScenes dataset, which maps the original categories into 10 foreground categories.
The baseline results include instance segmentation models, e.g., Mask R-CNN and Cascade Mask R-CNN.
We will support panoptic segmentation models in the future.


## Results

### Instance Segmentation

We report Mask R-CNN and Cascade Mask R-CNN results on nuimages.

|Method | Backbone|Pretraining | Lr schd | Mem (GB) | Box AP  | Mask AP  |Download |
| :---------: |:---------: | :---------: | :-----: |:-----: | :------: | :------------: | :----: |
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_1x_nuim.py) |IN|1x|7.4|47.8 |38.4|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_fpn_1x_nuim/mask_rcnn_r50_fpn_1x_nuim_20200906_114546-902bb808.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_fpn_1x_nuim/mask_rcnn_r50_fpn_1x_nuim_20200906_114546.log.json)|
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_coco-2x_1x_nuim.py) |IN+COCO-2x|1x|7.4|49.6|40.0|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_fpn_coco-2x_1x_nuim/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20200905_234546-01b6b9ba.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_fpn_coco-2x_1x_nuim/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20200905_234546.log.json)|
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_1x_nuim.py) |IN|1x|7.0|47.7|38.2|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_caffe_fpn_1x_nuim/mask_rcnn_r50_caffe_fpn_1x_nuim_20200906_120052-733905fa.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_caffe_fpn_1x_nuim/mask_rcnn_r50_caffe_fpn_1x_nuim_20200906_120052.log.json)|
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim.py) |IN+COCO-3x|1x|7.0|49.7|40.3|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim_20200906_134613-e6dc1931.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim_20200906_134613.log.json)|
| Mask R-CNN| [R-101](./mask_rcnn_r101_fpn_1x_nuim.py) |IN|1x|10.9|48.9|38.9|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r101_fpn_1x_nuim/mask_rcnn_r101_fpn_1x_nuim_20200906_182752-823be521.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_r101_fpn_1x_nuim/mask_rcnn_r101_fpn_1x_nuim_20200906_182752.log.json)|
| Mask R-CNN| [X-101_32x4d](./mask_rcnn_x101_32x4d_fpn_1x_nuim.py) |IN|1x|13.3|50.3|40.1|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_x101_32x4d_fpn_1x_nuim/mask_rcnn_x101_32x4d_fpn_1x_nuim_20200906_134611-bd241530.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/mask_rcnn_x101_32x4d_fpn_1x_nuim/mask_rcnn_x101_32x4d_fpn_1x_nuim_20200906_134611.log.json)|
| Cascade Mask R-CNN| [R-50](./cascade_mask_rcnn_r50_fpn_1x_nuim.py) |IN|1x|8.9|50.8|40.1|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/cascade_mask_rcnn_r50_fpn_1x_nuim/cascade_mask_rcnn_r50_fpn_1x_nuim_20200906_114546-22bf3085.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/cascade_mask_rcnn_r50_fpn_1x_nuim/cascade_mask_rcnn_r50_fpn_1x_nuim_20200906_114546.log.json)|
| Cascade Mask R-CNN| [R-101](./cascade_mask_rcnn_r101_fpn_1x_nuim.py) |IN|1x|12.5|51.8|40.6|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20200906_134611-ee279b07.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20200906_134611.log.json)|
| Cascade Mask R-CNN| [X-101_32x4d](./cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py) |IN|1x|14.9|52.9|41.6|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20200906_134611-47db31b0.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/nuimages/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20200906_134611.log.json)|

**Note**:
1. `IN` means only using ImageNet pre-trained backbone. `IN+COCO-Nx` means the backbone is first pre-trained on ImageNet, and then the detector is pre-trained on COCO train2017 dataset by `Nx` schedules.
2. All the training hyper-parameters follow the standard 1x schedules on COCO dataset except that the images are resized from
1280 x 720 to 1920 x 1080 (relative ratio 0.8 to 1.2) since the images are in size 1600 x 900.
