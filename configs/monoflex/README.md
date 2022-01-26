# Objects are Different: Flexible Monocular 3D Object Detection

## Abstract

<!-- [ABSTRACT] -->

The precise localization of 3D objects from a single image without depth information is a highly challenging problem. Most existing methods adopt the same approach for all objects regardless of their diverse distributions, leading to limited performance for truncated objects. In this paper, we propose a flexible framework for monocular 3D object detection which explicitly decouples the truncated objects and adaptively combines multiple approaches for object depth estimation. Specifically, we decouple the edge of the feature map for predicting long-tail truncated objects so that the optimization of normal objects is not influenced. Furthermore, we formulate the object depth estimation as an uncertainty-guided ensemble of directly regressed object depth and solved depths from different groups of keypoints. Experiments demonstrate that our method outperforms the state-of-the-art method by relatively 27% for the moderate level and 30% for the hard level in the test set of KITTI benchmark while maintaining real-time efficiency.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143886681-52cb72b9-6635-4624-a728-1c243b046517.png" width="800"/>
</div>

<!-- [PAPER_TITLE: Objects are Different: Flexible Monocular 3D Object Detection] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2104.02323] -->

## Introduction

<!-- [ALGORITHM] -->

We implement MonoFlex and provide the results and checkpoints on KITTI dataset.

```
@InProceedings{MonoFlex,
    author    = {Zhang, Yunpeng and Lu, Jiwen and Zhou, Jie},
    title     = {Objects Are Different: Flexible Monocular 3D Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3289-3298}
}
```

## Results

### KITTI

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | mAP | Download |
| :---------: | :-----: | :------: | :------------: | :----: | :------: |
|[DLA34](./monoflex_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py)|6x|9.64||21.86|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/monoflex/monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/monoflex/monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d_20210929_015553.log.json)

Note: mAP represents Car moderate 3D strict AP11 results.
Detailed performance on KITTI 3D detection (3D/BEV) is as follows, evaluated by AP11 and AP40 metric:

|             |     Easy      |    Moderate    |     Hard      |
|-------------|:-------------:|:--------------:|:-------------:|
| Car (AP11)  | 28.02 / 36.11 | 21.86 / 29.46  | 19.01 / 24.83 |
| Car (AP40)  | 23.22 / 32.74 | 17.18 / 24.02  | 15.13 / 20.67 |

Note: mAP represents Car moderate 3D strict AP11 / AP40 results. Because of the limited data for pedestrians and cyclists, the detection performance for these two classes is usually unstable. Therefore, we only list car detection results here. In addition, AP40 is a more recommended metric for reference due to its much better stability. Besides, the training of MonoFlex is unstable, the final result may fluctuate ~1 AP.
