# FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection

> [FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection](https://arxiv.org/abs/2112.00322)

<!-- [ALGORITHM] -->

## Abstract

Recently, promising applications in robotics and augmented reality have attracted considerable attention to 3D object detection from point clouds. In this paper, we present FCAF3D --- a first-in-class fully convolutional anchor-free indoor 3D object detection method. It is a simple yet effective method that uses a voxel representation of a point cloud and processes voxels with sparse convolutions. FCAF3D can handle large-scale scenes with minimal runtime through a single fully convolutional feed-forward pass. Existing 3D object detection methods make prior assumptions on the geometry of objects, and we argue that it limits their generalization ability. To eliminate prior assumptions, we propose a novel parametrization of oriented bounding boxes that allows obtaining better results in a purely data-driven way. The proposed method achieves state-of-the-art 3D object detection results in terms of mAP@0.5 on ScanNet V2 (+4.5), SUN RGB-D (+3.5), and S3DIS (+20.5) datasets.

<div align="center">
<img src="https://user-images.githubusercontent.com/6030962/182842796-98c10576-d39c-4c2b-a15a-a04c9870919c.png" width="800"/>
</div>

## Introduction

We implement FCAF3D and provide the result and checkpoints on the ScanNet dataset.
`Max` and `mean` metrics are copied from the paper and `ours` is for provided checkpoint.
`Mean` value is averaged across 5 train runs followed by 5 test runs.
Inference time is given for a single NVidia GTX1080ti GPU. All models are trained on 2 GPUs.

## Results and models

### ScanNet

|                      Backbone                      | Mem (GB) | Inf time (fps) |              AP@0.25              |              AP@0.5               | Download |
| :------------------------------------------------: | :------: | :------------: | :-------------------------------: | :-------------------------------: | :------: |
| [MinkResNet34](./fcaf3d_8x2_scannet-3d-18class.py) |   10.5   |      8.0       | max: 71.5, mean: 70.7, ours: 69.7 | max: 57.3, mean: 56.0, ours: 55.2 |    ?     |

### SUN RGB-D

|                      Backbone                      | Mem (GB) | Inf time (fps) |              AP@0.25              |              AP@0.5               | Download |
| :------------------------------------------------: | :------: | :------------: | :-------------------------------: | :-------------------------------: | :------: |
| [MinkResNet34](./fcaf3d_8x2_sunrgbd-3d-10class.py) |   6.3    |      15.6      | max: 64.2, mean: 63.8, ours: 64.8 | max: 48.9, mean: 48.2, ours: 48.2 |    ?     |

### S3DIS

|                    Backbone                     | Mem (GB) | Inf time (fps) |              AP@0.25              |              AP@0.5               | Download |
| :---------------------------------------------: | :------: | :------------: | :-------------------------------: | :-------------------------------: | :------: |
| [MinkResNet34](./fcaf3d_8x2_s3dis-3d-5class.py) |   23.5   |      4.2       | max: 66.7, mean: 64.9, ours: 67.4 | max: 45.9, mean: 43.8, ours: 45.7 |    ?     |

## Citation

```latex
@inproceedings{rukhovich2022fcaf3d,
  title={FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection},
  author={Danila Rukhovich, Anna Vorontsova, Anton Konushin},
  booktitle={European conference on computer vision},
  year={2022}
}
```
