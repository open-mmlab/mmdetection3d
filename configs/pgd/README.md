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
