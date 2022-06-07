# Dynamic Graph CNN for Learning on Point Clouds

> [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)

<!-- [ALGORITHM] -->

## Abstract

Point clouds provide a flexible geometric representation suitable for countless applications in computer graphics; they also comprise the raw output of most 3D data acquisition devices. While hand-designed features on point clouds have long been proposed in graphics and vision, however, the recent overwhelming success of convolutional neural networks (CNNs) for image analysis suggests the value of adapting insight from CNN to the point cloud world. Point clouds inherently lack topological information so designing a model to recover topology can enrich the representation power of point clouds. To this end, we propose a new neural network module dubbed EdgeConv suitable for CNN-based high-level tasks on point clouds including classification and segmentation. EdgeConv acts on graphs dynamically computed in each layer of the network. It is differentiable and can be plugged into existing architectures. Compared to existing modules operating in extrinsic space or treating each point independently, EdgeConv has several appealing properties: It incorporates local neighborhood information; it can be stacked applied to learn global shape properties; and in multi-layer systems affinity in feature space captures semantic characteristics over potentially long distances in the original embedding. We show the performance of our model on standard benchmarks including ModelNet40, ShapeNetPart, and S3DIS.

<div align=center>
<img src="https://user-images.githubusercontent.com/30491025/143855852-3d7888ed-2cfc-416c-9ec8-57621edeaa34.png" width="800"/>
</div>

## Introduction

We implement DGCNN and provide the results and checkpoints on S3DIS dataset.

**Notice**: We follow the implementations in the original DGCNN paper and a PyTorch implementation of DGCNN [code](https://github.com/AnTao97/dgcnn.pytorch).

## Results and models

### S3DIS

|                          Method                           | Split  |   Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) |                                                                                                                                                                                                 Download                                                                                                                                                                                                 |
| :-------------------------------------------------------: | :----: | :---------: | :------: | :------------: | :------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_1 | cosine 100e |   13.1   |                |     68.33      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area1/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_000734-39658f14.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area1/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_000734.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_2 | cosine 100e |   13.1   |                |     40.68      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area2/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_144648-aea9ecb6.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area2/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_144648.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_3 | cosine 100e |   13.1   |                |     69.38      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area3/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210801_154629-2ff50ee0.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area3/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210801_154629.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_4 | cosine 100e |   13.1   |                |     50.07      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area4/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_073551-dffab9cd.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area4/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_073551.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_5 | cosine 100e |   13.1   |                |     50.59      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area5/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210730_235824-f277e0c5.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area5/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210730_235824.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_6 | cosine 100e |   13.1   |                |     77.94      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area6/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_154317-e3511b32.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area6/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_154317.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | 6-fold |             |          |                |     59.43      |                                                                                                                                                                                                                                                                                                                                                                                                          |

**Notes:**

- We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
- `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.
- `6-fold` Split means the overall result of 6 different splits (Area_1, Area_2, Area_3, Area_4, Area_5 and Area_6 Splits).
- Users need to modify `train_area` and `test_area` in the S3DIS dataset's [config](./configs/_base_/datasets/s3dis_seg-3d-13class.py) to set the training and testing areas, respectively.

## Indeterminism

Since DGCNN testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.

## Citation

```latex
@article{dgcnn,
  title={Dynamic Graph CNN for Learning on Point Clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019}
}
```
