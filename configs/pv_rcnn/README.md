# PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection

> [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://arxiv.org/abs/1912.13192)

<!-- [ALGORITHM] -->

## Abstract

3D object detection has been receiving increasing attention from both industry and academia thanks to its wide applications in various fields such as autonomous driving and robotics. LiDAR sensors are widely adopted in autonomous driving vehicles and robots for capturing 3D scene information as sparse and irregular point clouds, which provide vital cues for 3D scene perception and understanding. In this paper, we propose to achieve high performance 3D object detection by designing novel point-voxel integrated networks to learn better 3D features from irregular point clouds.

<div align=center>
<img src="https://user-images.githubusercontent.com/88368822/202114244-ccf52f56-b8c9-4f1b-9cc2-80c7a9952c99.png" width="800"/>
</div>

## Results and models

### KITTI

|                    Backbone                     |  Class  |  Lr schd   | Mem (GB) | Inf time (fps) |  mAP  |                                                                                                                                                                    Download                                                                                                                                                                    |
| :---------------------------------------------: | :-----: | :--------: | :------: | :------------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [SECFPN](./pv_rcnn_8xb2-80e_kitti-3d-3class.py) | 3 Class | cyclic 80e |   5.4    |                | 72.28 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth) \\ [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428.json) |

Note: mAP represents AP11 results on 3 Class under the moderate setting.

Detailed performance on KITTI 3D detection (3D) is as follows, evaluated by AP11 metric:

|            | Easy  | Moderate | Hard  |
| ---------- | :---: | :------: | :---: |
| Car        | 89.20 |  83.72   | 78.79 |
| Pedestrian | 66.64 |  59.84   | 55.33 |
| Cyclist    | 87.25 |  73.27   | 69.61 |

## Citation

```latex
@article{ShaoshuaiShi2020PVRCNNPF,
  title={PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection},
  author={Shaoshuai Shi and Chaoxu Guo and Li Jiang and Zhe Wang and Jianping Shi and Xiaogang Wang and Hongsheng Li},
  journal={computer vision and pattern recognition},
  year={2020}
}
```
