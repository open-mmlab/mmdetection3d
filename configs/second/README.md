# Second: Sparsely embedded convolutional detection

## Introduction

<!-- [ALGORITHM] -->

We implement SECOND and provide the results and checkpoints on KITTI dataset.

```
@article{yan2018second,
  title={Second: Sparsely embedded convolutional detection},
  author={Yan, Yan and Mao, Yuxing and Li, Bo},
  journal={Sensors},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## Results

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP |Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [SECFPN](./hv_second_secfpn_6x8_80e_kitti-3d-car.py)| Car |cyclic 80e|5.4||79.07|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238.log.json)|
|    [SECFPN](./hv_second_secfpn_6x8_80e_kitti-3d-3class.py)| 3 Class |cyclic 80e|5.4||64.41|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238.log.json)|

### Waymo

|  Backbone | Load Interval | Class | Lr schd | Mem (GB) | Inf time (fps) | mAP@L1 | mAPH@L1 |  mAP@L2 | **mAPH@L2** | Download |
| :-------: | :-----------: |:-----:| :------:| :------: | :------------: | :----: | :-----: | :-----: | :-----: | :------: |
| [SECFPN](./hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)|5|3 Class|2x|8.12||65.3|61.7|58.9|55.7|[log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_sbn_4x8_2x_waymoD5-3d-3class/hv_second_secfpn_sbn_4x8_2x_waymoD5-3d-3class_20201115_112448.log.json)|
| above @ Car|||2x|8.12||67.1|66.6|58.7|58.2| |
| above @ Pedestrian|||2x|8.12||68.1|59.1|59.5|51.5| |
| above @ Cyclist|||2x|8.12||60.7|59.5|58.4|57.3| |

Note: See more details about metrics and data split on Waymo [HERE](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars). For implementation details, we basically follow the original settings. All of these results are achieved without bells-and-whistles, e.g. ensemble, multi-scale training and test augmentation.
