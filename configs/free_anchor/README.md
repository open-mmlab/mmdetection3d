# FreeAnchor for 3D Object Detection

> [FreeAnchor: Learning to Match Anchors for Visual Object Detection](https://arxiv.org/abs/1909.02466)

<!-- [ALGORITHM] -->

## Abstract

Modern CNN-based object detectors assign anchors for ground-truth objects under the restriction of object-anchor Intersection-over-Unit (IoU). In this study, we propose a learning-to-match approach to break IoU restriction, allowing objects to match anchors in a flexible manner. Our approach, referred to as FreeAnchor, updates hand-crafted anchor assignment to “free" anchor matching by formulating detector training as a maximum likelihood estimation (MLE) procedure. FreeAnchor targets at learning features which best explain a class of objects in terms of both classification and localization. FreeAnchor is implemented by optimizing detection customized likelihood and can be fused with CNN-based detectors in a plug-and-play manner. Experiments on COCO demonstrate that FreeAnchor consistently outperforms the counterparts with significant margins.

<div align=center>
<img src="https://user-images.githubusercontent.com/36950400/143866685-e3ac08bb-cd0c-4ada-ba8a-18e03cccdd0f.png" width="600"/>
</div>

## Introduction

We implement FreeAnchor in 3D detection systems and provide their first results with PointPillars on nuScenes dataset.
With the implemented `FreeAnchor3DHead`, a PointPillar detector with a big backbone (e.g., RegNet-3.2GF) achieves top performance
on the nuScenes benchmark.

## Usage

### Modify config

As in the [baseline config](hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py), we only need to replace the head of an existing one-stage detector to use FreeAnchor head.
Since the config is inherit from a common detector head, `_delete_=True` is necessary to avoid conflicts.
The hyperparameters are specifically tuned according to the original paper.

```python
_base_ = [
    '../_base_/models/hv_pointpillars_fpn_lyft.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pts_bbox_head=dict(
        _delete_=True,
        type='FreeAnchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg = dict(
        pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25])))
```

## Results and models

### PointPillars

|                                                 Backbone                                                  | FreeAnchor | Lr schd | Mem (GB) | Inf time (fps) |  mAP  |  NDS  |                                                                                                                                                                                                                                                                    Download                                                                                                                                                                                                                                                                    |
| :-------------------------------------------------------------------------------------------------------: | :--------: | :-----: | :------: | :------------: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                    [FPN](../pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)                    |     ✗      |   2x    |   17.1   |                | 40.0  | 53.3  |                                                                        [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405.log.json)                                                                        |
|                     [FPN](./hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)                     |     ✓      |   2x    |   16.3   |                | 43.82 | 54.86 |                                                 [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210816_163441-ae0897e7.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210816_163441.log.json)                                                 |
|         [RegNetX-400MF-FPN](../regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py)          |     ✗      |   2x    |   17.3   |                | 44.8  | 56.4  |                                                    [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20200620_230239-c694dce7.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20200620_230239.log.json)                                                    |
|       [RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)        |     ✓      |   2x    |   17.6   |                | 48.3  | 58.65 |                       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210827_213939-a2dd3fff.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210827_213939.log.json)                       |
|       [RegNetX-1.6GF-FPN](./hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)        |     ✓      |   2x    |   24.3   |                | 52.04 | 61.49 |                       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210828_025608-bfbd506e.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210828_025608.log.json)                       |
| [RegNetX-1.6GF-FPN](./hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py)\* |     ✓      |   3x    |   24.4   |                | 52.69 | 62.45 | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d_20210827_184909-14d2dbd1.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d_20210827_184909.log.json) |
|       [RegNetX-3.2GF-FPN](./hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)        |     ✓      |   2x    |   29.4   |                | 52.4  | 61.94 |                       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210827_181237-e385c35a.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210827_181237.log.json)                       |
| [RegNetX-3.2GF-FPN](./hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py)\* |     ✓      |   3x    |   29.2   |                | 54.23 | 63.41 | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d_20210828_030816-06708918.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d/hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d_20210828_030816.log.json) |

**Note**: Models noted by `*` means it is trained using stronger augmentation with vertical flip under bird-eye-view, global translation, and larger range of global rotation.

## Citation

```latex
@inproceedings{zhang2019freeanchor,
  title   =  {{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author  =  {Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle =  {Neural Information Processing Systems},
  year    =  {2019}
}
```
