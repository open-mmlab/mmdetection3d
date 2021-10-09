# 教程 4: 自定义模型

我们通常把模型的各个组成成分分成6种类型：

- 编码器（encoder）：包括 voxel layer、voxel encoder 和 middle encoder 等进入 backbone 前所使用的基于 voxel 的方法，如 HardVFE 和 PointPillarsScatter。
- 骨干网络（backbone）：通常采用 FCN 网络来提取特征图，如 ResNet 和 SECOND。
- 颈部网络（neck）：位于 backbones 和 heads 之间的组成模块，如 FPN 和 SECONDFPN。
- 检测头（head）：用于特定任务的组成模块，如检测框的预测和掩码的预测。
- RoI 提取器（RoI extractor）：用于从特征图中提取 RoI 特征的组成模块，如 H3DRoIHead 和 PartAggregationROIHead。
- 损失函数（loss）：heads 中用于计算损失函数的组成模块，如 FocalLoss、L1Loss 和 GHMLoss。

## 开发新的组成模块

### 添加新建 encoder

接下来我们以 HardVFE 为例展示如何开发新的组成模块。

#### 1. 定义一个新的 voxel encoder（如 HardVFE：即 DV-SECOND 中所提出的 Voxel 特征提取器）

创建一个新文件 `mmdet3d/models/voxel_encoders/voxel_encoder.py` ：

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

#### 2. 导入新建模块

用户可以通过添加下面这行代码到 `mmdet3d/models/voxel_encoders/__init__.py` 中

```python
from .voxel_encoder import HardVFE
```

或者添加以下的代码到配置文件中，从而能够在避免修改源码的情况下导入新建模块。

```python
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.HardVFE'],
    allow_failed_imports=False)
```

#### 3. 在配置文件中使用 voxel encoder

```python
model = dict(
    ...
    voxel_encoder=dict(
        type='HardVFE',
        arg1=xxx,
        arg2=xxx),
    ...
```

### 添加新建 backbone

接下来我们以 [SECOND](https://www.mdpi.com/1424-8220/18/10/3337)（Sparsely Embedded Convolutional Detection） 为例展示如何开发新的组成模块。

#### 1. 定义一个新的 backbone（如 SECOND）

创建一个新文件 `mmdet3d/models/backbones/second.py` ：

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

#### 2. 导入新建模块

用户可以通过添加下面这行代码到 `mmdet3d/models/backbones/__init__.py` 中

```python
from .second import SECOND
```

或者添加以下的代码到配置文件中，从而能够在避免修改源码的情况下导入新建模块。

```python
custom_imports = dict(
    imports=['mmdet3d.models.backbones.second'],
    allow_failed_imports=False)
```

#### 3. 在配置文件中使用 backbone

```python
model = dict(
    ...
    backbone=dict(
        type='SECOND',
        arg1=xxx,
        arg2=xxx),
    ...
```

### 添加新建 necks

#### 1. 定义一个新的 neck（如 SECONDFPN）

创建一个新文件 `mmdet3d/models/necks/second_fpn.py` ：

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

#### 2. 导入新建模块

用户可以通过添加下面这行代码到 `mmdet3D/models/necks/__init__.py` 中

```python
from .second_fpn import SECONDFPN
```

或者添加以下的代码到配置文件中，从而能够在避免修改源码的情况下导入新建模块。

```python
custom_imports = dict(
    imports=['mmdet3d.models.necks.second_fpn'],
    allow_failed_imports=False)
```

#### 3. 在配置文件中使用 neck

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

### 添加新建 heads

接下来我们以 [PartA2 Head](https://arxiv.org/abs/1907.03670) 为例展示如何开发新的组成模块。

**注意**：此处展示的 PartA2 RoI Head 将应用于双阶段检测器中，对于单阶段检测器，请参考 `mmdet3d/models/dense_heads/` 中所展示的例子。由于这些 heads 简单高效，因此这些 heads 普遍应用在自动驾驶场景下的 3D 检测任务中。

首先，在 `mmdet3d/models/roi_heads/bbox_heads/parta2_bbox_head.py` 中创建一个新的 bbox head。
PartA2 RoI Head 实现一个新的 bbox head ，并用于目标检测的任务中。
为了实现一个新的 bbox head，通常需要在其中实现三个功能，如下所示，有时该模块还需要实现其他相关的功能，如 `loss` 和 `get_targets`。

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

其次，如果有必要的话，用户还需要实现一个新的 RoI Head，此处我们从 `Base3DRoIHead` 中继承得到一个新类 `PartAggregationROIHead`，此时我们就能发现 `Base3DRoIHead` 已经实现了下面的功能：

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

接着将会对 bbox_forward 的逻辑进行修改，同时，bbox_forward 还会继承来自 `Base3DRoIHead` 的其他逻辑，在 `mmdet3d/models/roi_heads/part_aggregation_roi_head.py` 中，我们实现了新的 RoI Head，如下所示：

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

此处我们省略了与其他功能相关的细节，请参考 [此处](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/roi_heads/part_aggregation_roi_head.py) 获取更多细节。

最后，用户需要在 `mmdet3d/models/bbox_heads/__init__.py` 和 `mmdet3d/models/roi_heads/__init__.py` 中添加新模块，使得对应的注册器能够发现并加载该模块。

此外，用户也可以添加以下的代码到配置文件中，从而实现相同的目标。

```python
custom_imports=dict(
    imports=['mmdet3d.models.roi_heads.part_aggregation_roi_head', 'mmdet3d.models.roi_heads.bbox_heads.parta2_bbox_head'])
```

PartAggregationROIHead 的配置文件如下所示：

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

MMDetection 2.0 支持配置文件之间的继承，使得用户能够更加关注自己的配置文件的修改。
PartA2 Head 的第二阶段主要使用新建的 `PartAggregationROIHead` 和 `PartA2BboxHead`，需要根据对应模块的 `__init__` 参数来设置对应的参数。

### 添加新建 loss

假定用户想要新添一个用于检测框回归的 loss，并命名为 `MyLoss`。
为了添加一个新的 loss ，用于需要在 `mmdet3d/models/losses/my_loss.py` 中实现对应的逻辑。
装饰器 `weighted_loss` 能够保证对 batch 中每个样本的 loss 进行加权平均。

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

接着，用户需要将 loss 添加到 `mmdet3d/models/losses/__init__.py`：

```python
from .my_loss import MyLoss, my_loss

```

此外，用户也可以添加以下的代码到配置文件中，从而实现相同的目标。

```python
custom_imports=dict(
    imports=['mmdet3d.models.losses.my_loss'])
```

为了使用该 loss，需要对 `loss_xxx` 域进行修改。
因为 MyLoss 主要用于检测框的回归，因此需要在对应的 head 中修改 `loss_bbox` 域的值。

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
