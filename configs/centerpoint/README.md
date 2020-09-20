# Center-based 3D Object Detection and Tracking

## Introduction

We implement CenterPoint and provide the result and checkpoints on nuScenes dataset.

We follow the below style to name config files. Contributors are advised to follow the same style.
`{xxx}` is required field and `[yyy]` is optional.

`{model}`: model type like `centerpoint`.

`{model setting}`: voxel size and voxel type like `01voxel`, `02pillar`.

`{backbone}`: backbone type like `second`.

`{neck}`: neck type like `secfpn`.

`[dcn]`: Whether to use deformable convolution.

`[circle]`: Whether to use circular nms.

`[batch_per_gpu x gpu]`: GPUs and samples per GPU, 4x8 is used by default.

`{schedule}`: training schedule, options are 1x, 2x, 20e, etc. 1x and 2x means 12 epochs and 24 epochs respectively. 20e is adopted in cascade models, which denotes 20 epochs. For 1x/2x, initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs. For 20e, initial learning rate decays by a factor of 10 at the 16th and 19th epochs.

`{dataset}`: dataset like nus-3d, kitti-3d, lyft-3d, scannet-3d, sunrgbd-3d. We also indicate the number of classes we are using if there exist multiple settings, e.g., kitti-3d-3class and kitti-3d-car means training on KITTI dataset with 3 classes and single class, respectively.
```
@article{yin2020center,
  title={Center-based 3d object detection and tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2006.11275},
  year={2020}
}
```

## Results

### CenterPoint

|Backbone|  Voxel type (voxel size)   |Dcn|Circular nms| Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: |:-----: |:-----: | :------: | :------------: | :----: |:----: | :------: |:------: |
|[SECFPN](./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✗||||||
|[SECFPN](./centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✓||||||
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✗||||||
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✓||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✗||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✓||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✗||||||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✓||||||
|[SECFPN](./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✗||||||
|[SECFPN](./centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✓|||48.72|59.40||
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✗||||||
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✓||||||
