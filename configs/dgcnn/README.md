# Dynamic Graph CNN for Learning on Point Clouds

## Introduction

<!-- [ALGORITHM] -->

We implement DGCNN and provide the result and checkpoints on ScanNet and S3DIS datasets.

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
| [DGCNN](./dgcnn_32x1_cosine_100e_s3dis_seg-3d-13class.py) | Area_5 | cosine 100e |   6.7    |                |     46.96      | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dgcnn/dgcnn_32x1_cosine_100e_s3dis_seg-3d-13class/dgcnn_32x1_cosine_100e_s3dis_seg-3d-13class_20210514_143628-4e341a48.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dgcnn/dgcnn_32x1_cosine_100e_s3dis_seg-3d-13class/dgcnn_32x1_cosine_100e_s3dis_seg-3d-13class_20210514_143628.log.json) |

**Notes:**

-   We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
-   `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.

## Indeterminism

Since DGCNN testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.
