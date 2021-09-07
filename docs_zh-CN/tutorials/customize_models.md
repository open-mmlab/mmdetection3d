# 教程 4: 自定义模型

我们通常把模型的各个组成成分分成6种类型：

- encoder：包括 voxel layer、voxel encoder 和 middle encoder 等进入 backbone 前所使用的基于 voxel 的方法，如 HardVFE 和 PointPillarsScatter。
- backbone：通常采用 FCN 网络来提取特征图，如 ResNet 和 SECOND。
- neck：位于 backbones 和 heads 之间的组成模块，如 FPN 和 SECONDFPN。
- head：用于特定任务的组成模块，如检测框的预测和掩码的预测。
- roi extractor：用于从特征图中提取 RoI 特征的组成模块，如 H3DRoIHead 和 PartAggregationROIHead。
- loss：heads 中用于计算损失函数的组成模块，如 FocalLoss、L1Loss 和 GHMLoss。

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

其次，如果有必要的话，用户还需要实现一个新的 RoI Head，此处我们从 `Base3DRoIHead` 中继承得到一个新类 `PartAggregationROIHead` ，此时我们就能发现 `Base3DRoIHead` 已经实现了下面的功能：

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
