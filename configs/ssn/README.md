# SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds

## Introduction

<!-- [ALGORITHM] -->

We implement PointPillars with Shape-aware grouping heads used in the SSN and provide the results and checkpoints on the nuScenes and Lyft dataset.

```
@inproceedings{zhu2020ssn,
  title={SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds},
  author={Zhu, Xinge and Ma, Yuexin and Wang, Tai and Xu, Yan and Shi, Jianping and Lin, Dahua},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
```

## Results

### NuScenes

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP | NDS | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||35.17|49.76|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725.log.json)|
|[SSN](./hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py)|2x|9.62||41.56|54.83|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20201023_193737-5fda3f00.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20201023_193737.log.json)|
[RegNetX-400MF-SECFPN](../regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.4||41.15|55.20|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334-53044f32.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334.log.json)|
|[RegNetX-400MF-SSN](./hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d.py)|2x|10.26||46.95|58.24|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20201024_232447-7af3d8c8.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20201024_232447.log.json)|

### Lyft

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d.py)|2x|12.2||13.9|14.1|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807-2518e3de.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807.log.json)|
|[SSN](./hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py)|2x|8.30||17.4|17.5|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d_20201016_220844-3058d9fc.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d_20201016_220844.log.json)|
|[RegNetX-400MF-SSN](./hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d.py)|2x|9.98||18.1|18.3|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_lyft-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_lyft-3d_20201025_213155-4532096c.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_lyft-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_lyft-3d_20201025_213155.log.json)|

Note:

The main difference of the shape-aware grouping heads with the original SECOND FPN heads is that the former groups objects with similar sizes and shapes together, and design shape-specific heads for each group. Heavier heads (with more convolutions and large strides) are designed for large objects while smaller heads for small objects. Note that there may appear different feature map sizes in the outputs, so an anchor generator tailored to these feature maps is also needed in the implementation.

Users could try other settings in terms of the head design. Here we basically refer to the implementation [HERE](https://github.com/xinge008/SSN).
