# PointPillars: Fast Encoders for Object Detection from Point Clouds

## Introduction

<!-- [ALGORITHM] -->

We implement PointPillars and provide the results and checkpoints on KITTI, nuScenes, Lyft and Waymo datasets.

```
@inproceedings{lang2019pointpillars,
  title={Pointpillars: Fast encoders for object detection from point clouds},
  author={Lang, Alex H and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12697--12705},
  year={2019}
}

```

## Results

### KITTI

|  Backbone|Class   | Lr schd | Mem (GB) | Inf time (fps) | AP  |Download |
| :---------: | :-----: |:-----: | :------: | :------------: | :----: | :------: |
|    [SECFPN](./hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py)|Car|cyclic 160e|5.4||77.1|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-77663cd6.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614.log.json)|
|    [SECFPN](./hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py)|3 Class|cyclic 160e|5.5||59.5|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421.log.json)|

### nuScenes

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](./hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.2||34.33|49.08|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857.log.json)|
|[FPN](./hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)|2x|16.3||39.71|53.15|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936.log.json)|

### Lyft

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score | Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|[SECFPN](./hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d.py)|2x|12.2||13.8|14.1|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210829_100455-82b81c39.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210829_100455.log.json)|
|[SECFPN-Range100](./hv_pointpillars_secfpn_sbn-all_range100_2x8_2x_lyft-3d.py)|2x|18.5||14.8|15.0|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_range100_2x8_2x_lyft-3d_20210829_204415-2043eef3.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_secfpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_range100_2x8_2x_lyft-3d_20210829_204415.log.json)|
|[FPN](./hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py)|2x|9.3||14.8|15.0|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210822_095429-0b3d6196.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210822_095429.log.json)|
|[FPN-Range100](./hv_pointpillars_fpn_sbn-all_range100_2x8_2x_lyft-3d.py)|2x|13.8||15.5|15.7|[model](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_range100_2x8_2x_lyft-3d_20210829_053451-3ad1d9fc.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.0.0/models/pointpillars/hv_pointpillars_fpn_sbn-all_range100_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_range100_2x8_2x_lyft-3d_20210829_053451.log.json)|

### Waymo

|  Backbone | Load Interval | Class | Lr schd | Mem (GB) | Inf time (fps) | mAP@L1 | mAPH@L1 |  mAP@L2 | **mAPH@L2** | Download |
| :-------: | :-----------: |:-----:| :------:| :------: | :------------: | :----: | :-----: | :-----: | :-----: | :------: |
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car.py)|5|Car|2x|7.76||70.2|69.6|62.6|62.1|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315.log.json)|
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py)|5|3 Class|2x|8.12||64.7|57.6|58.4|52.1|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144-d1a706b1.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144.log.json)|
| above @ Car|||2x|8.12||68.5|67.9|60.1|59.6| |
| above @ Pedestrian|||2x|8.12||67.8|50.6|59.6|44.3| |
| above @ Cyclist|||2x|8.12||57.7|54.4|55.5|52.4| |
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car.py)|1|Car|2x|7.76||72.1|71.5|63.6|63.1|[log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car.log.json)|
| [SECFPN](./hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class.py)|1|3 Class|2x|8.12||68.8|63.3|62.6|57.6|[log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class.log.json)|
| above @ Car|||2x|8.12||71.6|71.0|63.1|62.5| |
| above @ Pedestrian|||2x|8.12||70.6|56.7|62.9|50.2| |
| above @ Cyclist|||2x|8.12||64.4|62.3|61.9|59.9| |

#### Note:

- **Metric**: For model trained with 3 classes, the average APH@L2 (mAPH@L2) of all the categories is reported and used to rank the model. For model trained with only 1 class, the APH@L2 is reported and used to rank the model.
- **Data Split**: Here we provide several baselines for waymo dataset, among which D5 means that we divide the dataset into 5 folds and only use one fold for efficient experiments. Using the complete dataset can boost the performance a lot, especially for the detection of cyclist and pedestrian, where more than 5 mAP or mAPH improvement can be expected.
- **Implementation Details**: We basically follow the implementation in the [paper](https://arxiv.org/pdf/1912.04838.pdf) in terms of the network architecture (having a
stride of 1 for the first convolutional block). Different settings of voxelization, data augmentation and hyper parameters make these baselines outperform those in the paper by about 7 mAP for car and 4 mAP for pedestrian with only a subset of the whole dataset. All of these results are achieved without bells-and-whistles, e.g. ensemble, multi-scale training and test augmentation.
- **License Aggrement**: To comply the [license agreement of Waymo dataset](https://waymo.com/open/terms/), the pre-trained models on Waymo dataset are not released. We still release the training log as a reference to ease the future research.
