# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import ConfigType
from .decode_head import Base3DDecodeHead

if IS_TORCHSPARSE_AVAILABLE:
    from torchsparse import SparseTensor


@MODELS.register_module()
class MinkUNetHead(Base3DDecodeHead):
    """
    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
    """

    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(channels, num_classes, **kwargs)
        self.conv_seg = nn.Linear(channels, num_classes)

    def predict(self, inputs: SparseTensor, data_samples: SampleList,
                test_cfg: ConfigType) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level point features.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.
            test_cfg (dict): The testing config.

        Returns:
            list[Tensor]: The segmentation prediction mask of each batch.
        """
        seg_logits = self.forward(inputs)
        seg_preds = seg_logits.argmax(dim=1)

        batch_idx = inputs.C[:, -1]
        seg_pred_list = []
        for i, data_sample in enumerate(data_samples):
            seg_pred = seg_preds[batch_idx == i]
            seg_pred = seg_pred[data_sample.gt_pts_seg.point2voxel_map]
            seg_pred_list.append(seg_pred)

        return seg_pred_list

    def forward(self, x: SparseTensor) -> Tensor:
        """Forward function.

        Args:
            x (SparseTensor): Features from backbone with shape [N, C].

        Returns:
            output (Tensor): Segmentation map of shape [N, C].
                Note that output contains all points from each batch.
        """
        output = self.cls_seg(x.F)
        return output
