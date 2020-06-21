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
|    [SECFPN](./hv_second_secfpn_6x8_80e_kitti-3d-car.py)| Car |cyclic 80e|5.4||79.07|
|    [SECFPN](./hv_second_secfpn_6x8_80e_kitti-3d-3class.py)| 3 Class |cyclic 80e|5.4||64.41|
