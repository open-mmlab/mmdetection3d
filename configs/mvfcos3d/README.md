# MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones

> [MV-FCOS3D++: Multi-View} Camera-Only 4D Object Detection with Pretrained Monocular Backbones](https://arxiv.org/abs/2207.12716)

<!-- [ALGORITHM] -->

## Abstract

In this technical report, we present our solution, dubbed MV-FCOS3D++, for the Camera-Only 3D Detection track in Waymo Open Dataset Challenge 2022. For multi-view camera-only 3D detection, methods based on bird-eye-view or 3D geometric representations can leverage the stereo cues from overlapped regions between adjacent views and directly perform 3D detection without hand-crafted post-processing. However, it lacks direct semantic supervision for 2D backbones, which can be complemented by pretraining simple monocular-based detectors. Our solution is a multi-view framework for 4D detection following this paradigm. It is built upon a simple monocular detector FCOS3D++, pretrained only with object annotations of Waymo, and converts multi-view features to a 3D grid space to detect 3D objects thereon. A dual-path neck for single-frame understanding and temporal stereo matching is devised to incorporate multi-frame information. Our method finally achieves 49.75% mAPL with a single model and wins 2nd place in the WOD challenge, without any LiDAR-based depth supervision during training. The code will be released at [this https URL](https://github.com/Tai-Wang/Depth-from-Motion).

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection3d/assets/72679458/9313eb3c-cc41-40be-9ead-549b3b5fef44" width="800"/>
</div>

## Introduction

We implement multi-view FCOS3D++ and provide the results on Waymo dataset.

## Usage

### Training commands

1. You should train PGD first:

```bash
bash tools/dist_train.py configs/pgd/pgd_r101_fpn_gn-head_dcn_8xb3-2x_waymoD3-mv-mono3d.py 8
```

2. Given pre-trained PGD backbone, you could train multi-view FCOS3D++:

```bash
bash tools/dist_train.sh configs/mvfcos3d/multiview-fcos3d_r101-dcn_8xb2_waymoD5-3d-3class.py --cfg-options load_from=${PRETRAINED_CHECKPOINT}
```

**Note**:
the path of `load_from` needs to be changed to yours accordingly.

## Results and models

### Waymo

|                                Backbone                                | Load Interval | mAPL | mAP  | mAPH |                                                                                             Download                                                                                             |
| :--------------------------------------------------------------------: | :-----------: | :--: | :--: | :--: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [ResNet101+DCN](./multiview-fcos3d_r101-dcn_8xb2_waymoD5-3d-3class.py) |      5x       | 38.2 | 52.9 | 49.5 | [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvfcos3d/multiview-fcos3d_r101-dcn_8xb2_waymoD5-3d-3class/multiview-fcos3d_r101-dcn_8xb2_waymoD5-3d-3class_20231127_122815.log) |
|                              above @ Car                               |               | 56.5 | 73.3 | 72.3 |                                                                                                                                                                                                  |
|                           above @ Pedestrian                           |               | 34.8 | 49.5 | 43.1 |                                                                                                                                                                                                  |
|                            above @ Cyclist                             |               | 23.2 | 35.9 | 33.3 |                                                                                                                                                                                                  |

**Note**:

Regrettably, we are unable to provide the pre-trained model weights due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), so we only provide the training logs as shown above.

## Citation

```latex
@article{wang2022mvfcos3d++,
  title={{MV-FCOS3D++: Multi-View} Camera-Only 4D Object Detection with Pretrained Monocular Backbones},
  author={Wang, Tai and Lian, Qing and Zhu, Chenming and Zhu, Xinge and Zhang, Wenwei},
  journal={arXiv preprint},
  year={2022}
}
```
