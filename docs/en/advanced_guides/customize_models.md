# Customize Models

We basically categorize model components into 6 types:

- encoder: Including voxel encoder and middle encoder used in voxel-based methods before backbone, e.g., `HardVFE` and `PointPillarsScatter`.
- backbone: Usually an FCN network to extract feature maps, e.g., `ResNet`, `SECOND`.
- neck: The component between backbones and heads, e.g., `FPN`, `SECONDFPN`.
- head: The component for specific tasks, e.g., `bbox prediction` and `mask prediction`.
- RoI extractor: The part for extracting RoI features from feature maps, e.g., `H3DRoIHead` and `PartAggregationROIHead`.
- loss: The component in heads for calculating losses, e.g., `FocalLoss`, `L1Loss`, and `GHMLoss`.

## Develop new components

### Add a new encoder

Here we show how to develop new components with an example of HardVFE.

#### 1. Define a new voxel encoder (e.g. HardVFE: Voxel feature encoder used in HV-SECOND)

Create a new file `mmdet3d/models/voxel_encoders/voxel_encoder.py`.

```python
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class HardVFE(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. Import the module

You can either add the following line to `mmdet3d/models/voxel_encoders/__init__.py`:

```python
from .voxel_encoder import HardVFE
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.voxel_encoder'],
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
        arg2=yyy),
    ...
)
```

### Add a new backbone

Here we show how to develop new components with an example of [SECOND](https://www.mdpi.com/1424-8220/18/10/3337) (Sparsely Embedded Convolutional Detection).

#### 1. Define a new backbone (e.g. SECOND)

Create a new file `mmdet3d/models/backbones/second.py`.

```python
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SECOND(BaseModule):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. Import the module

You can either add the following line to `mmdet3d/models/backbones/__init__.py`:

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
        arg2=yyy),
    ...
)
```

### Add a new neck

#### 1. Define a new neck (e.g. SECONDFPN)

Create a new file `mmdet3d/models/necks/second_fpn.py`.

```python
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
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

    def forward(self, x):
        # implementation is ignored
        pass
```

#### 2. Import the module

You can either add the following line to `mmdet3d/models/necks/__init__.py`:

```python
from .second_fpn import SECONDFPN
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmdet3d.models.necks.second_fpn'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

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
)
```

### Add a new head

Here we show how to develop a new head with the example of [PartA2 Head](https://arxiv.org/abs/1907.03670) as the following.

**Note**: Here the example of `PartA2 RoI Head` is used in the second stage. For one-stage heads, please refer to examples in `mmdet3d/models/dense_heads/`. They are more commonly used in 3D detection for autonomous driving due to its simplicity and high efficiency.

First, add a new bbox head in `mmdet3d/models/roi_heads/bbox_heads/parta2_bbox_head.py`.
`PartA2 RoI Head` implements a new bbox head for object detection.
To implement a bbox head, basically we need to implement two functions of the new module as the following. Sometimes other related functions like `loss` and `get_targets` are also required.

```python
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
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
        pass
```

Second, implement a new RoI Head if it is necessary. We plan to inherit the new `PartAggregationROIHead` from `Base3DRoIHead`. We can find that a `Base3DRoIHead` already implements the following functions.

```python
from mmdet.models.roi_heads import BaseRoIHead

from mmdet3d.registry import MODELS, TASK_UTILS


class Base3DRoIHead(BaseRoIHead):
    """Base class for 3d RoIHeads."""

    def __init__(self,
                 bbox_head=None,
                 bbox_roi_extractor=None,
                 mask_head=None,
                 mask_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Base3DRoIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_head=mask_head,
            mask_roi_extractor=mask_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    TASK_UTILS.build(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)

    def init_mask_head(self):
        """Initialize mask head, skip since ``PartAggregationROIHead`` does not
        have one."""
        pass
```

Double Head's modification is mainly in the bbox_forward logic, and it inherits other logics from the `Base3DRoIHead`.
In the `mmdet3d/models/roi_heads/part_aggregation_roi_head.py`, we implement the new RoI Head as the following:

```python
from typing import Dict, List, Tuple

from mmdet.models.task_modules import AssignResult, SamplingResult
from mmengine import ConfigDict
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.utils import InstanceList
from ...structures.det3d_data_sample import SampleList
from .base_3droi_head import Base3DRoIHead


@MODELS.register_module()
class PartAggregationROIHead(Base3DRoIHead):
    """Part aggregation roi head for PartA2.

    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        bbox_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 semantic_head: dict,
                 num_classes: int = 3,
                 seg_roi_extractor: dict = None,
                 bbox_head: dict = None,
                 bbox_roi_extractor: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None) -> None:
        super(PartAggregationROIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        assert semantic_head is not None
        self.init_seg_head(seg_roi_extractor, semantic_head)

    def init_seg_head(self, seg_roi_extractor: dict,
                      semantic_head: dict) -> None:
        """Initialize semantic head and seg roi extractor.

        Args:
            seg_roi_extractor (dict): Config of seg
                roi extractor.
            semantic_head (dict): Config of semantic head.
        """
        self.semantic_head = MODELS.build(semantic_head)
        self.seg_roi_extractor = MODELS.build(seg_roi_extractor)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def predict(self,
                feats_dict: Dict,
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        assert self.with_bbox, 'Bbox head must be implemented in PartA2.'
        assert self.with_semantic, 'Semantic head must be implemented' \
                                   ' in PartA2.'

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        voxels_dict = feats_dict.pop('voxels_dict')
        # TODO: Split predict semantic and bbox
        results_list = self.predict_bbox(feats_dict, voxels_dict,
                                         batch_input_metas, rpn_results_list,
                                         self.test_cfg)
        return results_list

    def predict_bbox(self, feats_dict: Dict, voxel_dict: Dict,
                     batch_input_metas: List[dict],
                     rpn_results_list: InstanceList,
                     test_cfg: ConfigDict) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxel_dict (dict): Contains information of voxels.
            batch_input_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            test_cfg (Config): Test config.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        ...

    def loss(self, feats_dict: Dict, rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        losses = dict()
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        voxels_dict = feats_dict.pop('voxels_dict')
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict, voxels_dict, batch_gt_instances_3d)
            losses.update(semantic_results.pop('loss_semantic'))

        sample_results = self._assign_and_sample(rpn_results_list,
                                                 batch_gt_instances_3d)
        if self.with_bbox:
            feats_dict.update(semantic_results)
            bbox_results = self._bbox_forward_train(feats_dict, voxels_dict,
                                                    sample_results)
            losses.update(bbox_results['loss_bbox'])

        return losses
```

Here we omit more details related to other functions. Please see the [code](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/models/roi_heads/part_aggregation_roi_head.py) for more details.

Last, the users need to add the module in
`mmdet3d/models/roi_heads/bbox_heads/__init__.py` and `mmdet3d/models/roi_heads/__init__.py` thus the corresponding registry could find and load them.

Alternatively, the users can add

```python
custom_imports=dict(
    imports=['mmdet3d.models.roi_heads.part_aggregation_roi_head', 'mmdet3d.models.roi_heads.bbox_heads.parta2_bbox_head'],
    allow_failed_imports=False)
```

to the config file and achieve the same goal.

The config file of `PartAggregationROIHead` is as the following:

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
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_part=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        bbox_roi_extractor=dict(
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
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    ...
)
```

Since MMDetection 2.0, the config system supports to inherit configs such that the users can focus on the modification.
The second stage of PartA2 Head mainly uses a new `PartAggregationROIHead` and a new
`PartA2BboxHead`, the arguments are set according to the `__init__` function of each module.

### Add a new loss

Assume you want to add a new loss as `MyLoss` for bounding box regression.
To add a new loss function, the users need to implement it in `mmdet3d/models/losses/my_loss.py`.
The decorator `weighted_loss` enables the loss to be weighted for each element.

```python
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from mmdet3d.registry import MODELS


@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@MODELS.register_module()
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
    imports=['mmdet3d.models.losses.my_loss'],
    allow_failed_imports=False)
```

to the config file and achieve the same goal.

To use it, users should modify the `loss_xxx` field.
Since `MyLoss` is for regression, you need to modify the `loss_bbox` field in the head.

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0)
```
