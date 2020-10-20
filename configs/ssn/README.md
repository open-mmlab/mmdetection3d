# SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds

## Introduction

We implement PointPillars with Shape-aware grouping heads used in the SSN and provide the results and checkpoints on Lyft datasets.

```
@inproceedings{zhu2020ssn,
  title={SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds},
  author={Zhu, Xinge and Ma, Yuexin and Wang, Tai and Xu, Yan and Shi, Jianping and Lin, Dahua},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}

```

## Results

### Lyft

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](../pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_lyft-3d.py)|2x|||13.4|13.4||
|[SSN](./hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py)|2x|8.30||17.4|17.5||

Note:

The main difference of the shape-aware grouping heads with the original SECOND FPN heads is that the former groups objects with similar sizes and shapes together, and design shape-specific heads for each group. Heavier heads (with more convolutions and large strides) are designed for large objects while smaller heads for small objects. Note that there may appear different feature map sizes in the outputs, so an anchor generator tailored to these feature maps is also needed in the implementation.

Users could try other settings in terms of the head design. Here we basically refer to the implementation [HERE](https://github.com/xinge008/SSN).
