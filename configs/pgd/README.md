# Probabilistic and Geometric Depth: Detecting Objects in Perspective

## Introduction

<!-- [ALGORITHM] -->

PGD, also can be regarded as FCOS3D++, is a simple yet effective monocular 3D detector. It enhances the FCOS3D baseline by involving local geometric constraints and improving instance depth estimation.

We first release the code and model for KITTI benchmark, which is a good supplement for the original FCOS3D baseline (only supported on nuScenes). Models for nuScenes will be released soon.

For clean implementation, our preliminary release supports base models with proposed local geometric constraints and the probabilistic depth representation. We will involve the geometric graph part in the future.

```
@inproceedings{wang2021pgd,
    title={Probabilistic and Geometric Depth: Detecting Objects in Perspective},
    author={Wang, Tai and Zhu, Xinge and Pang, Jiangmiao and Lin, Dahua},
    booktitle={Conference on Robot Learning (CoRL) 2021},
    year={2021}
}
```

## Results

### KITTI

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP_11 / mAP_40 | Download |
| :---------: | :-----: | :------: | :------------: | :----: | :------: |
|[ResNet101](./pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py)|4x|9.07||18.33 / 13.23|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608.log.json)

Detailed performance on KITTI 3D detection (3D/BEV) is as follows, evaluated by AP11 and AP40 metric:

|             |     Easy      |    Moderate    |     Hard      |
|-------------|:-------------:|:--------------:|:-------------:|
| Car (AP11)  | 24.09 / 30.11 | 18.33 / 23.46  | 16.90 / 19.33 |
| Car (AP40)  | 19.27 / 26.60 | 13.23 / 18.23  | 10.65 / 15.00 |

Note: mAP represents Car moderate 3D strict AP11 / AP40 results. Because of the limited data for pedestrians and cyclists, the detection performance for these two classes is usually unstable. Therefore, we only list car detection results here. In addition, AP40 is a more recommended metric for reference due to its much better stability.
