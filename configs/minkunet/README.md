# 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks

> [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755)

<!-- [ALGORITHM] -->

## Abstract

In many robotics and VR/AR applications, 3D-videos are readily-available sources of input (a continuous sequence of depth images, or LIDAR scans). However, those 3D-videos are processed frame-by-frame either through 2D convnets or 3D perception algorithms. In this work, we propose 4-dimensional convolutional neural networks for spatio-temporal perception that can directly process such 3D-videos using high-dimensional convolutions. For this, we adopt sparse tensors and propose the generalized sparse convolution that encompasses all discrete convolutions. To implement the generalized sparse convolution, we create an open-source auto-differentiation library for sparse tensors that provides extensive functions for high-dimensional convolutional neural networks. We create 4D spatio-temporal convolutional neural networks using the library and validate them on various 3D semantic segmentation benchmarks and proposed 4D datasets for 3D-video perception. To overcome challenges in the 4D space, we propose the hybrid kernel, a special case of the generalized sparse convolution, and the trilateral-stationary conditional random field that enforces spatio-temporal consistency in the 7D space-time-chroma space. Experimentally, we show that convolutional neural networks with only generalized 3D sparse convolutions can outperform 2D or 2D-3D hybrid methods by a large margin. Also, we show that on 3D-videos, 4D spatio-temporal convolutional neural networks are robust to noise, outperform 3D convolutional neural networks and are faster than the 3D counterpart in some cases.

<div align=center>
<img src="https://user-images.githubusercontent.com/72679458/225243534-cd0ed738-4224-4e7c-bcac-4f4c8d89f3a9.png" width="800"/>
</div>

## Introduction

We implement MinkUNet with [TorchSparse](https://github.com/mit-han-lab/torchsparse) backend and provide the result and checkpoints on SemanticKITTI datasets.

## Results and models

### SemanticKITTI

|    Method    | Lr schd | Mem (GB) | mIoU |                                                                                                                                                                          Download                                                                                                                                                                           |
| :----------: | :-----: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MinkUNet-W16 |   15e   |   3.4    | 60.3 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w16_8xb2-15e_semantickitti/minkunet_w16_8xb2-15e_semantickitti_20230309_160737-0d8ec25b.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w16_8xb2-15e_semantickitti/minkunet_w16_8xb2-15e_semantickitti_20230309_160737.log) |
| MinkUNet-W20 |   15e   |   3.7    | 61.6 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w20_8xb2-15e_semantickitti/minkunet_w20_8xb2-15e_semantickitti_20230309_160718-c3b92e6e.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w20_8xb2-15e_semantickitti/minkunet_w20_8xb2-15e_semantickitti_20230309_160718.log) |
| MinkUNet-W32 |   15e   |   4.9    | 63.1 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w32_8xb2-15e_semantickitti/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w32_8xb2-15e_semantickitti/minkunet_w32_8xb2-15e_semantickitti_20230309_160710.log) |

**Note:** We follow the implementation in SPVNAS original [repo](https://github.com/mit-han-lab/spvnas) and W16\\W20\\W32 indicates different number of channels.

**Note:** Due to TorchSparse backend, the model performance is unstable with TorchSparse backend and may fluctuate by about 1.5 mIoU for different random seeds.

## Citation

```latex
@inproceedings{choy20194d,
  title={4d spatio-temporal convnets: Minkowski convolutional neural networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={3075--3084},
  year={2019}
}
```
