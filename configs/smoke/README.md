# SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation

> [SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation](https://arxiv.org/abs/2002.10111)

<!-- [ALGORITHM] -->

## Abstract

Estimating 3D orientation and translation of objects is essential for infrastructure-less autonomous navigation and driving. In case of monocular vision, successful methods have been mainly based on two ingredients: (i) a network generating 2D region proposals, (ii) a R-CNN structure predicting 3D object pose by utilizing the acquired regions of interest. We argue that the 2D detection network is redundant and introduces non-negligible noise for 3D detection. Hence, we propose a novel 3D object detection method, named SMOKE, in this paper that predicts a 3D bounding box for each detected object by combining a single keypoint estimate with regressed 3D variables. As a second contribution, we propose a multi-step disentangling approach for constructing the 3D bounding box, which significantly improves both training convergence and detection accuracy. In contrast to previous 3D detection techniques, our method does not require complicated pre/post-processing, extra data, and a refinement stage. Despite of its structural simplicity, our proposed SMOKE network outperforms all existing monocular 3D detection methods on the KITTI dataset, giving the best state-of-the-art result on both 3D object detection and Bird's eye view evaluation.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143886681-52cb72b9-6635-4624-a728-1c243b046517.png" width="800"/>
</div>

## Introduction

We implement SMOKE and provide the results and checkpoints on KITTI dataset.

## Results and models

### KITTI

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: | :------: | :------------: | :----: | :------: |
|[DLA34](./smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py)|6x|9.64||13.85|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553.log.json)

Note: mAP represents Car moderate 3D strict AP11 results.

Detailed performance on KITTI 3D detection (3D/BEV) is as follows, evaluated by AP11 metric:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 16.92 / 22.97 | 13.85 / 18.32  | 11.90 / 15.88|
| Pedestrian  | 11.13 / 12.61| 11.10 / 11.32  | 10.67 / 11.14|
| Cyclist     | 0.99  / 1.47  | 0.54 / 0.65    | 0.55 / 0.67  |

## Citation

```latex
@inproceedings{liu2020smoke,
  title={Smoke: Single-stage monocular 3d object detection via keypoint estimation},
  author={Liu, Zechen and Wu, Zizhang and T{\'o}th, Roland},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={996--997},
  year={2020}
}
```
