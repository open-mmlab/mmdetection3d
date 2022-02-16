# PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds

> [PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds](https://arxiv.org/abs/2103.14635)

<!-- [ALGORITHM] -->

## Abstract

We introduce Position Adaptive Convolution (PAConv), a generic convolution operation for 3D point cloud processing. The key of PAConv is to construct the convolution kernel by dynamically assembling basic weight matrices stored in Weight Bank, where the coefficients of these weight matrices are self-adaptively learned from point positions through ScoreNet. In this way, the kernel is built in a data-driven manner, endowing PAConv with more flexibility than 2D convolutions to better handle the irregular and unordered point cloud data. Besides, the complexity of the learning process is reduced by combining weight matrices instead of brutally predicting kernels from point positions.
Furthermore, different from the existing point convolution operators whose network architectures are often heavily engineered, we integrate our PAConv into classical MLP-based point cloud pipelines without changing network configurations. Even built on simple networks, our method still approaches or even surpasses the state-of-the-art models, and significantly improves baseline performance on both classification and segmentation tasks, yet with decent efficiency. Thorough ablation studies and visualizations are provided to understand PAConv.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143881915-003d5f10-3999-474e-969a-c354cb738a11.png" width="800"/>
</div>

## Introduction

We implement PAConv and provide the result and checkpoints on S3DIS dataset.

**Notice**: The original PAConv paper used step learning rate schedule. We discovered that cosine schedule achieves slightly better results and adopt it in our implementations.

## Results and models

### S3DIS

|                                   Method                                    | Split  |   Lr schd   | Mem (GB) | Inf time (fps) | mIoU (Val set) |         Download         |
| :-------------------------------------------------------------------------: | :----: | :---------: | :------: | :------------: | :------------: | :----------------------: |
|    [PAConv (SSG)](./paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class.py)     | Area_5 | cosine 150e |   5.8    |                |       66.65        | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class_20210729_200615-2147b2d1.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class_20210729_200615.log.json) |
|    [PAConv\* (SSG)](./paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py)     | Area_5 | cosine 200e |   3.8    |                |       65.33        | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class_20210802_171802-e5ea9bb9.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class_20210802_171802.log.json) |

**Notes:**

- We use XYZ+Color+Normalized_XYZ as input in all the experiments on S3DIS datasets.
- `Area_5` Split means training the model on Area_1, 2, 3, 4, 6 and testing on Area_5.
- PAConv\* stands for the CUDA implementation of PAConv operations. See the [paper](https://arxiv.org/pdf/2103.14635.pdf) appendix section D for more details. In our experiments, the training of PAConv\* is found to be very unstable. We achieved slightly lower mIoU than the result in the paper, but is consistent with the result obtained by running their [official code](https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg). Besides, although the GPU memory consumption of PAConv\* is significantly lower than PAConv, its training and inference speed are actually slower (by ~10%).

## Indeterminism

Since PAConv testing adopts sliding patch inference which involves random point sampling, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.

## Citation

```latex
@inproceedings{xu2021paconv,
  title={PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
  author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3173--3182},
  year={2021}
}
```
