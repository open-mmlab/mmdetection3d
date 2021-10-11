# Dynamic Graph CNN for Learning on Point Clouds

## Introduction

<!-- [ALGORITHM] -->

We implement DGCNN and provide the results and checkpoints on S3DIS dataset.

```
@article{dgcnn,
  title={Dynamic Graph CNN for Learning on Point Clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019}
}
```

**Notice**: We follow the implementations in the original DGCNN paper and a PyTorch implementation of DGCNN [code](https://github.com/AnTao97/dgcnn.pytorch).

## Results

### S3DIS

|                                   Method                                    | Split  |  Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) |         Download         |
| :-------------------------------------------------------------------------: | :----: | :--------: | :------: | :------------: | :------------: | :----------------------: |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_1 | cosine 100e |   13.1    |                |     68.33      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area1/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_000734-39658f14.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area1/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_000734.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_2 | cosine 100e |   13.1    |                |     40.68      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area2/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_144648-aea9ecb6.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area2/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_144648.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_3 | cosine 100e |   13.1    |                |     69.38      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area3/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210801_154629-2ff50ee0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area3/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210801_154629.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_4 | cosine 100e |   13.1    |                |     50.07      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area4/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_073551-dffab9cd.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area4/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_073551.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_5 | cosine 100e |   13.1    |                |     50.59      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area5/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210730_235824-f277e0c5.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area5/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210730_235824.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | Area_6 | cosine 100e |   13.1    |                |     77.94      | [model](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area6/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_154317-e3511b32.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area6/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210802_154317.log.json) |
| [DGCNN](./dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py) | 6-fold |           |           |                |     59.43      |        |

**Notes:**

-   We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
-   `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.
-   `6-fold` Split means the overall result of 6 different splits (Area_1, Area_2, Area_3, Area_4, Area_5 and Area_6 Splits).
-   Users need to modify `train_area` and `test_area` in the S3DIS dataset's [config](./configs/_base_/datasets/s3dis_seg-3d-13class.py) to set the training and testing areas, respectively.

## Indeterminism

Since DGCNN testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.
