# SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds

> [SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds](https://arxiv.org/abs/2004.02774)

<!-- [ALGORITHM] -->

## Abstract

Multi-class 3D object detection aims to localize and classify objects of multiple categories from point clouds. Due to the nature of point clouds, i.e. unstructured, sparse and noisy, some features benefit-ting multi-class discrimination are underexploited, such as shape information. In this paper, we propose a novel 3D shape signature to explore the shape information from point clouds. By incorporating operations of symmetry, convex hull and chebyshev fitting, the proposed shape sig-nature is not only compact and effective but also robust to the noise, which serves as a soft constraint to improve the feature capability of multi-class discrimination. Based on the proposed shape signature, we develop the shape signature networks (SSN) for 3D object detection, which consist of pyramid feature encoding part, shape-aware grouping heads and explicit shape encoding objective. Experiments show that the proposed method performs remarkably better than existing methods on two large-scale datasets. Furthermore, our shape signature can act as a plug-and-play component and ablation study shows its effectiveness and good scalability.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/144024507-9c1f23c1-5e5a-49c8-b346-ff37e30adc3a.png" width="800"/>
</div>

## Introduction

We implement PointPillars with Shape-aware grouping heads used in the SSN and provide the results and checkpoints on the nuScenes and Lyft dataset.

## Results and models

### NuScenes

|                                            Backbone                                            | Lr schd | Mem (GB) | Inf time (fps) |  mAP  |  NDS  |                                                                                                                                                                                                                       Download                                                                                                                                                                                                                       |
| :--------------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|           [SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)            |   2x    |   16.4   |                | 35.17 | 49.76 |                     [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725.log.json)                     |
|                        [SSN](./hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py)                        |   2x    |   3.6    |                | 40.91 | 54.44 |                                              [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351.log.json)                                              |
| [RegNetX-400MF-SECFPN](../regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d.py) |   2x    |   16.4   |                | 41.15 | 55.20 | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334-53044f32.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334.log.json) |
|          [RegNetX-400MF-SSN](./hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d.py)           |   2x    |   5.1    |                | 46.65 | 58.24 |                    [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615.log.json)                    |

### Lyft

|                                   Backbone                                   | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score |                                                                                                                                                                                                      Download                                                                                                                                                                                                      |
| :--------------------------------------------------------------------------: | :-----: | :------: | :------------: | :-----------: | :----------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d.py)  |   2x    |   12.2   |                |     13.9      |     14.1     |  [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807-2518e3de.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807.log.json)  |
|              [SSN](./hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py)               |   2x    |   8.5    |                |     17.5      |     17.5     |                           [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d_20210822_134731-46841b41.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d_20210822_134731.log.json)                           |
| [RegNetX-400MF-SSN](./hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d.py) |   2x    |   7.4    |                |     17.9      |      18      | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d_20210829_122825-d93475a1.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d_20210829_122825.log.json) |

Note:

The main difference of the shape-aware grouping heads with the original SECOND FPN heads is that the former groups objects with similar sizes and shapes together, and design shape-specific heads for each group. Heavier heads (with more convolutions and large strides) are designed for large objects while smaller heads for small objects. Note that there may appear different feature map sizes in the outputs, so an anchor generator tailored to these feature maps is also needed in the implementation.

Users could try other settings in terms of the head design. Here we basically refer to the implementation [HERE](https://github.com/xinge008/SSN).

## Citation

```latex
@inproceedings{zhu2020ssn,
  title={SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds},
  author={Zhu, Xinge and Ma, Yuexin and Wang, Tai and Xu, Yan and Shi, Jianping and Lin, Dahua},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
```
