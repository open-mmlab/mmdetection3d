# Second: Sparsely embedded convolutional detection

## Introduction

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
| [SECFPN](./hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)|5|3 Class|2x|8.12||58.0|53.5|51.5|48.3|[log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_sbn_4x8_2x_waymoD5-3d-3class/hv_second_secfpn_sbn_4x8_2x_waymoD5-3d-3class_20201115_112448.log.json)|
| above @ Car|||2x|8.12||58.5|57.9|51.6|51.1| |
| above @ Pedestrian|||2x|8.12||63.9|54.9|56.0|48.0| |
| above @ Cyclist|||2x|8.12||48.6|47.6|46.8|45.8| |

Note: See more details about metrics and data split on Waymo [HERE](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars). For implementation details, we basically follow the original settings. All of these results are achieved without bells-and-whistles, e.g. ensemble, multi-scale training and test augmentation.
