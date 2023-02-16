# FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection

> [FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection](https://arxiv.org/abs/2112.00322)

<!-- [ALGORITHM] -->

## Abstract

Recently, promising applications in robotics and augmented reality have attracted considerable attention to 3D object detection from point clouds. In this paper, we present FCAF3D --- a first-in-class fully convolutional anchor-free indoor 3D object detection method. It is a simple yet effective method that uses a voxel representation of a point cloud and processes voxels with sparse convolutions. FCAF3D can handle large-scale scenes with minimal runtime through a single fully convolutional feed-forward pass. Existing 3D object detection methods make prior assumptions on the geometry of objects, and we argue that it limits their generalization ability. To eliminate prior assumptions, we propose a novel parametrization of oriented bounding boxes that allows obtaining better results in a purely data-driven way. The proposed method achieves state-of-the-art 3D object detection results in terms of mAP@0.5 on ScanNet V2 (+4.5), SUN RGB-D (+3.5), and S3DIS (+20.5) datasets.

<div align="center">
<img src="https://user-images.githubusercontent.com/6030962/182842796-98c10576-d39c-4c2b-a15a-a04c9870919c.png" width="800"/>
</div>

## Introduction

We implement FCAF3D and provide the result and checkpoints on the ScanNet and SUN RGB-D dataset.

## Results and models

### ScanNet

|                      Backbone                      | Mem (GB) | Inf time (fps) |   AP@0.25    |    AP@0.5    |                                                                                                                                                          Download                                                                                                                                                           |
| :------------------------------------------------: | :------: | :------------: | :----------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./fcaf3d_8x2_scannet-3d-18class.py) |   10.5   |      15.7      | 69.7(70.7\*) | 55.2(56.0\*) | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_scannet-3d-18class/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_scannet-3d-18class/fcaf3d_8x2_scannet-3d-18class_20220805_084956.log.json) |

### SUN RGB-D

|                      Backbone                      | Mem (GB) | Inf time (fps) |   AP@0.25    |    AP@0.5    |                                                                                                                                                          Download                                                                                                                                                           |
| :------------------------------------------------: | :------: | :------------: | :----------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./fcaf3d_8x2_sunrgbd-3d-10class.py) |   6.3    |      17.9      | 63.8(63.8\*) | 47.3(48.2\*) | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_sunrgbd-3d-10class/fcaf3d_8x2_sunrgbd-3d-10class_20220805_165017.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_sunrgbd-3d-10class/fcaf3d_8x2_sunrgbd-3d-10class_20220805_165017.log.json) |

### S3DIS

|                     Backbone                     | Mem (GB) | Inf time (fps) |   AP@0.25    |    AP@0.5    |                                                                                                                                                    Download                                                                                                                                                     |
| :----------------------------------------------: | :------: | :------------: | :----------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./fcaf3d_2xb8_s3dis-3d-5class.py) |   23.5   |      10.9      | 67.4(64.9\*) | 45.7(43.8\*) | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_s3dis-3d-5class/fcaf3d_8x2_s3dis-3d-5class_20220805_121957.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_s3dis-3d-5class/fcaf3d_8x2_s3dis-3d-5class_20220805_121957.log.json) |

**Note**

- We report the results across 5 train runs followed by 5 test runs. * means the results reported in the paper.
- Inference time is given for a single NVidia RTX 4090 GPU. All models are trained on 2 GPUs.

## Citation

```latex
@inproceedings{rukhovich2022fcaf3d,
  title={FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection},
  author={Danila Rukhovich, Anna Vorontsova, Anton Konushin},
  booktitle={European conference on computer vision},
  year={2022}
}
```
