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
|[SECFPN](./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✗| | |56.56|64.46||
|[SECFPN](./centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✗|✓|4.9| |56.19|64.43|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205.log.json)|
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✗| | |56.60|64.90||
|[SECFPN](./centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.1)|✓|✓|5.2| |56.34|64.81|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20201004_075317-26d8176c.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20201004_075317.log.json)|
|[SECFPN](./centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✗| | |57.63|65.39| |
|[SECFPN](./centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✗|✓|7.8| |57.34|65.23|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20200925_230905-358fbe3b.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20200925_230905.log.json)|
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✗| | |57.43|65.63||
|[SECFPN](./centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|voxel (0.075)|✓|✓|8.5| |57.27|65.58|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619.log.json)|
|[SECFPN](./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✗| | |49.12|59.66||
|[SECFPN](./centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✗|✓|4.4| |49.07|59.66|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716.log.json)|
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✗| 4.6| |48.8 |59.67 |[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20200930_103722-3bb135f2.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20200930_103722.log.json)|
|[SECFPN](./centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py)|pillar (0.2)|✓|✓| | |48.79|59.65||
