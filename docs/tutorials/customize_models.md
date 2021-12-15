# Tutorial 4: Customize Models

We basically categorize model components into 6 types.

- encoder: including voxel layer, voxel encoder and middle encoder used in voxel-based methods before backbone, e.g., HardVFE and PointPillarsScatter.
- backbone: usually an FCN network to extract feature maps, e.g., ResNet, SECOND.
- neck: the component between backbones and heads, e.g., FPN, SECONDFPN.
- head: the component for specific tasks, e.g., bbox prediction and mask prediction.
- RoI extractor: the part for extracting RoI features from feature maps, e.g., H3DRoIHead and PartAggregationROIHead.
- loss: the component in heads for calculating losses, e.g., FocalLoss, L1Loss, and GHMLoss.

## Develop new components

### Add a new encoder

Here we show how to develop new components with an example of HardVFE.

#### 1. Define a new voxel encoder (e.g. HardVFE: Voxel feature encoder used in DV-SECOND)

Create a new file `mmdet3d/models/voxel_encoders/voxel_encoder.py`.

```python
import torch.nn as nn

from ..builder import VOXEL_ENCODERS


@VOXEL_ENCODERS.register_module()
class HardVFE(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. Import the module

You can either add the following line to `mmdet3d/models/voxel_encoders/__init__.py`

```python
from .voxel_encoder import HardVFE
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.HardVFE'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

#### 3. Use the voxel encoder in your config file

```python
model = dict(
    ...
    voxel_encoder=dict(
        type='HardVFE',
        arg1=xxx,
        arg2=xxx),
    ...
```

### Add a new backbone

Here we show how to develop new components with an example of [SECOND](https://www.mdpi.com/1424-8220/18/10/3337) (Sparsely Embedded Convolutional Detection).

#### 1. Define a new backbone (e.g. SECOND)

Create a new file `mmdet3d/models/backbones/second.py`.

```python
import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class SECOND(BaseModule):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. Import the module

You can either add the following line to `mmdet3d/models/backbones/__init__.py`

```python
from .second import SECOND
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmdet3d.models.backbones.second'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

#### 3. Use the backbone in your config file

```python
model = dict(
    ...
    backbone=dict(
        type='SECOND',
        arg1=xxx,
        arg2=xxx),
    ...
```

### Add new necks

#### 1. Define a neck (e.g. SECONDFPN)

Create a new file `mmdet3d/models/necks/second_fpn.py`.

```python
from ..builder import NECKS

@NECKS.register
class SECONDFPN(BaseModule):

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        pass

    def forward(self, X):
        # implementation is ignored
        pass
```

#### 2. Import the module

You can either add the following line to `mmdet3D/models/necks/__init__.py`,

```python
from .second_fpn import SECONDFPN
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmdet3d.models.necks.second_fpn'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Use the neck in your config file

```python
model = dict(
    ...
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    ...
```

### Add new heads

Here we show how to develop a new head with the example of [PartA2 Head](https://arxiv.org/abs/1907.03670) as the following.

**Note**: Here the example of PartA2 RoI Head is used in the second stage. For one-stage heads, please refer to examples in `mmdet3d/models/dense_heads/`. They are more commonly used in 3D detection for autonomous driving due to its simplicity and high efficiency.

First, add a new bbox head in `mmdet3d/models/roi_heads/bbox_heads/parta2_bbox_head.py`.
PartA2 RoI Head implements a new bbox head for object detection.
To implement a bbox head, basically we need to implement three functions of the new module as the following. Sometimes other related functions like `loss` and `get_targets` are also required.

```python
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead

@HEADS.register_module()
class PartA2BboxHead(BaseModule):
    """PartA2 RoI head."""

    def __init__(self,
                 num_classes,
                 seg_in_channels,
                 part_in_channels,
                 seg_conv_channels=None,
                 part_conv_channels=None,
                 merge_conv_channels=None,
                 down_conv_channels=None,
                 shared_fc_channels=None,
                 cls_channels=None,
                 reg_channels=None,
                 dropout_ratio=0.1,
                 roi_feat_size=14,
                 with_corner_loss=True,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg=None):
        super(PartA2BboxHead, self).__init__(init_cfg=init_cfg)

    def forward(self, seg_feats, part_feats):

```

Second, implement a new RoI Head if it is necessary. We plan to inherit the new `PartAggregationROIHead` from `Base3DRoIHead`. We can find that a `Base3DRoIHead` already implements the following functions.

```python
from abc import ABCMeta, abstractmethod
from torch import nn as nn


@HEADS.register_module()
class Base3DRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for 3d RoIHeads."""

    def __init__(self,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

    @property
    def with_bbox(self):

    @property
    def with_mask(self):

    @abstractmethod
    def init_weights(self, pretrained):

    @abstractmethod
    def init_bbox_head(self):

    @abstractmethod
    def init_mask_head(self):

    @abstractmethod
    def init_assigner_sampler(self):

    @abstractmethod
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass

```

Double Head's modification is mainly in the bbox_forward logic, and it inherits other logics from the `Base3DRoIHead`.
In the `mmdet3d/models/roi_heads/part_aggregation_roi_head.py`, we implement the new RoI Head as the following:

```python
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class PartAggregationROIHead(Base3DRoIHead):
    """Part aggregation roi head for PartA2.
    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        part_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 semantic_head,
                 num_classes=3,
                 seg_roi_extractor=None,
                 part_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(PartAggregationROIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        assert semantic_head is not None
        self.semantic_head = build_head(semantic_head)

        if seg_roi_extractor is not None:
            self.seg_roi_extractor = build_roi_extractor(seg_roi_extractor)
        if part_roi_extractor is not None:
            self.part_roi_extractor = build_roi_extractor(part_roi_extractor)

        self.init_assigner_sampler()

    def _bbox_forward(self, seg_feats, part_feats, voxels_dict, rois):
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.
        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.
            voxels_dict (dict): Contains information of voxels.
            rois (Tensor): Roi boxes.
        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_seg_feats = self.seg_roi_extractor(seg_feats,
                                                  voxels_dict['voxel_centers'],
                                                  voxels_dict['coors'][..., 0],
                                                  rois)
        pooled_part_feats = self.part_roi_extractor(
            part_feats, voxels_dict['voxel_centers'],
            voxels_dict['coors'][..., 0], rois)
        cls_score, bbox_pred = self.bbox_head(pooled_seg_feats,
                                              pooled_part_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            pooled_seg_feats=pooled_seg_feats,
            pooled_part_feats=pooled_part_feats)
        return bbox_results
```

Here we omit more details related to other functions. Please see the [code](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/roi_heads/part_aggregation_roi_head.py) for more details.

Last, the users need to add the module in
`mmdet3d/models/bbox_heads/__init__.py` and `mmdet3d/models/roi_heads/__init__.py` thus the corresponding registry could find and load them.

Alternatively, the users can add

```python
custom_imports=dict(
    imports=['mmdet3d.models.roi_heads.part_aggregation_roi_head', 'mmdet3d.models.roi_heads.bbox_heads.parta2_bbox_head'])
```

to the config file and achieve the same goal.

The config file of PartAggregationROIHead is as the following

```python
model = dict(
    ...
    roi_head=dict(
        type='PartAggregationROIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=16,
            extra_width=0.2,
            seg_score_thr=0.3,
            num_classes=3,
            loss_seg=dict(
                type='FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_part=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        part_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='avg')),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=3,
            seg_in_channels=16,
            part_in_channels=4,
            seg_conv_channels=[64, 64],
            part_conv_channels=[64, 64],
            merge_conv_channels=[128, 128],
            down_conv_channels=[128, 256],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            shared_fc_channels=[256, 512, 512, 512],
            cls_channels=[256, 256],
            reg_channels=[256, 256],
            dropout_ratio=0.1,
            roi_feat_size=14,
            with_corner_loss=True,
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0)))
    ...
    )
```

Since MMDetection 2.0, the config system supports to inherit configs such that the users can focus on the modification.
The second stage of PartA2 Head mainly uses a new `PartAggregationROIHead` and a new
`PartA2BboxHead`, the arguments are set according to the `__init__` function of each module.

### Add new loss

Assume you want to add a new loss as `MyLoss`, for bounding box regression.
To add a new loss function, the users need implement it in `mmdet3d/models/losses/my_loss.py`.
The decorator `weighted_loss` enable the loss to be weighted for each element.

```python
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
```

Then the users need to add it in the `mmdet3d/models/losses/__init__.py`.

```python
from .my_loss import MyLoss, my_loss

```

Alternatively, you can add

```python
custom_imports=dict(
    imports=['mmdet3d.models.losses.my_loss'])
```

to the config file and achieve the same goal.

To use it, modify the `loss_xxx` field.
Since MyLoss is for regression, you need to modify the `loss_bbox` field in the head.

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
