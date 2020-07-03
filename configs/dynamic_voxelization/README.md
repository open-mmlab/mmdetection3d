# Dynamic Voxelization

## Introduction

We implement Dynamic Voxelization proposed in  and provide its results and models on KITTI dataset.
```
@article{zhou2019endtoend,
    title={End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds},
    author={Yin Zhou and Pei Sun and Yu Zhang and Dragomir Anguelov and Jiyang Gao and Tom Ouyang and James Guo and Jiquan Ngiam and Vijay Vasudevan},
    year={2019},
    eprint={1910.06528},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```

## Results

### KITTI

|  Model   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: | :------: |
|[SECOND](./dv_second_secfpn_6x8_80e_kitti-3d-car.py)|Car    |cyclic 80e|5.5||78.83|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection3d/v0.1.0_models/) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection3d/v0.1.0_models/)|
|[SECOND](./dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py)| 3 Class|cosine 80e|5.5||65.10|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection3d/v0.1.0_models/) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection3d/v0.1.0_models/)|
|[PointPillars](./dv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py)| Car|cyclic 80e|4.7||77.76|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection3d/v0.1.0_models/) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection3d/v0.1.0_models/)|
