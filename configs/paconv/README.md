# PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds

## Introduction

<!-- [ALGORITHM] -->

We implement PAConv and provide the result and checkpoints on S3DIS dataset.

```
@inproceedings{xu2021paconv,
  title={PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
  author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3173--3182},
  year={2021}
}
```

**Notice**: The original PAConv paper used step learning rate schedule. We discovered that cosine schedule achieves slightly better results and adopt it in our implementations.

## Results

### S3DIS

|                                   Method                                    | Split  |   Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) |         Download         |
| :-------------------------------------------------------------------------: | :----: | :---------: | :------: | :------------: | :------------: | :----------------------: |
|    [PAConv (SSG)](./paconv_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py)     | Area_5 | cosine 200e |   5.8    |                |       -        | [model]() &#124; [log]() |
| [PAConv\* (SSG)](./paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py) | Area_5 | cosine 200e |   3.8    |                |       -        | [model]() &#124; [log]() |

**Notes:**

-   PAConv\* stands for the CUDA implementation of PAConv operations. See the [paper](https://arxiv.org/pdf/2103.14635.pdf) appendix section D for more details.
-   We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
-   `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.

## Indeterminism

Since PAConv testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.
