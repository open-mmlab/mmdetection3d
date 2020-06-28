# FreeAnchor for 3D Object Detection

## Introduction

We implement FreeAnchor in 3D detection systems and provide their first results with PointPillars on nuScenes dataset.
With the implemented `FreeAnchor3DHead`, a PointPillar detector with a big backbone (e.g., RegNet-3.2GF) achieves top performance
on the nuScenes benchmark.

```
@inproceedings{zhang2019freeanchor,
  title   =  {{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author  =  {Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle =  {Neural Information Processing Systems},
  year    =  {2019}
}
```

## Usage

### Modify config

As in the [baseline config](hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py), we only need to replace the head of an existing one-stage detector to use FreeAnchor head.
Since the config is inherit from a common detector head, `_delete_=True` is necessary to avoid conflicts.
The hyperparameters are specifically tuned according to the original paper.

```python
_base_ = [
    '../_base_/models/pointpillars_second_fpn.py',
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
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))
# model training and testing settings
train_cfg = dict(
    pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25]))
```

## Results

### PointPillars

|  Backbone   |FreeAnchor|Lr schd | Mem (GB) | Inf time (fps) | mAP |NDS| Download |
| :---------: |:-----: |:-----: | :------: | :------------: | :----: |:----: | :------: |
|[FPN](../pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py)|✗|2x|17.1||40.0|53.3||
|[FPN](./hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)|✓|2x|||43.7|55.1||
|[RegNetX-400MF-FPN](../regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py)|✗|2x|17.3||44.8|56.4||
|[RegNetX-400MF-FPN](./hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)|✓|2x||||||
|[RegNetX-1.6GF-FPN](./hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)|✓|2x||||||
|[RegNetX-3.2GF-FPN](./hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py)|✓|2x||||||
