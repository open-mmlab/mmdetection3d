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
|[SECOND](./dv_second_secfpn_6x8_80e_kitti-3d-car.py)|Car    |cyclic 80e|5.5||78.83||
|[SECOND](./dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py)| 3 Class|cosine 80e|5.5||65.10||
|[PointPillars](./dv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py)| Car|cyclic 80e|4.7||77.76||
