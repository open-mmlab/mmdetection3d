# Second: Sparsely embedded convolutional detection

> [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337)

<!-- [ALGORITHM] -->

## Abstract

LiDAR-based or RGB-D-based object detection is used in numerous applications, ranging from autonomous driving to robot vision. Voxel-based 3D convolutional networks have been used for some time to enhance the retention of information when processing point cloud LiDAR data. However, problems remain, including a slow inference speed and low orientation estimation performance. We therefore investigate an improved sparse convolution method for such networks, which significantly increases the speed of both training and inference. We also introduce a new form of angle loss regression to improve the orientation estimation performance and a new data augmentation approach that can enhance the convergence speed and performance. The proposed network produces state-of-the-art results on the KITTI 3D object detection benchmarks while maintaining a fast inference speed.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143889364-10be11c3-838e-4fc9-9613-184f0cd08907.png" width="800"/>
</div>

## Introduction

We implement SECOND and provide the results and checkpoints on KITTI dataset.

## Results and models

### KITTI

|                              Backbone                               |  Class  |  Lr schd   | Mem (GB) | Inf time (fps) |  mAP  |                                                                                                                                                                                             Download                                                                                                                                                                                             |
| :-----------------------------------------------------------------: | :-----: | :--------: | :------: | :------------: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        [SECFPN](./second_hv_secfpn_8xb6-80e_kitti-3d-car.py)        |   Car   | cyclic 80e |   5.4    |                | 78.2  |                       [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/second/second_hv_secfpn_8xb6-80e_kitti-3d-car/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/second/second_hv_secfpn_8xb6-80e_kitti-3d-car/second_hv_secfpn_8xb6-80e_kitti-3d-car-20230420_191750.log)                        |
|  [SECFPN (FP16)](./second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py)   |   Car   | cyclic 80e |   2.9    |                | 78.72 |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car_20200924_211301-1f5ad833.pth)\| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car_20200924_211301.log.json)        |
|      [SECFPN](./second_hv_secfpn_8xb6-80e_kitti-3d-3class.py)       | 3 Class | cyclic 80e |   5.4    |                | 65.3  |                 [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class/second_hv_secfpn_8xb6-80e_kitti-3d-3class-20230420_221130.log)                  |
| [SECFPN (FP16)](./second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py) | 3 Class | cyclic 80e |   2.9    |                | 67.4  | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059.log.json) |

### Waymo

|                              Backbone                              | Load Interval |  Class  | Lr schd | Mem (GB) | Inf time (fps) | mAP@L1 | mAPH@L1 | mAP@L2 | **mAPH@L2** |                                                                                           Download                                                                                            |
| :----------------------------------------------------------------: | :-----------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :----: | :---------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [SECFPN](./second_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py) |       5       | 3 Class |   2x    |   8.12   |                |  65.3  |  61.7   |  58.9  |    55.7     | [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_sbn_4x8_2x_waymoD5-3d-3class/hv_second_secfpn_sbn_4x8_2x_waymoD5-3d-3class_20201115_112448.log.json) |
|                            above @ Car                             |               |         |   2x    |   8.12   |                |  67.1  |  66.6   |  58.7  |    58.2     |                                                                                                                                                                                               |
|                         above @ Pedestrian                         |               |         |   2x    |   8.12   |                |  68.1  |  59.1   |  59.5  |    51.5     |                                                                                                                                                                                               |
|                          above @ Cyclist                           |               |         |   2x    |   8.12   |                |  60.7  |  59.5   |  58.4  |    57.3     |                                                                                                                                                                                               |

Note:

- See more details about metrics and data split on Waymo [HERE](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars). For implementation details, we basically follow the original settings. All of these results are achieved without bells-and-whistles, e.g. ensemble, multi-scale training and test augmentation.
- `FP16` means Mixed Precision (FP16) is adopted in training.

## Citation

```latex
@article{yan2018second,
  title={Second: Sparsely embedded convolutional detection},
  author={Yan, Yan and Mao, Yuxing and Li, Bo},
  journal={Sensors},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
