# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead

if IS_TORCHSPARSE_AVAILABLE:
    from torchsparse import SparseTensor
else:
    SparseTensor = None


@MODELS.register_module()
class MinkUNetHead(Base3DDecodeHead):
    r"""MinkUNet decoder head with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        channels (int): The input channel of conv_seg.
        num_classes (int): Number of classes.
    """

    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(channels, num_classes, **kwargs)
        self.conv_seg = nn.Linear(channels, num_classes)

    def loss(self, inputs: SparseTensor, data_samples: SampleList) -> dict:
        """Forward function for training.

        Args:
            inputs (SparseTensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        targets = torch.cat(
            [i.gt_pts_seg.voxel_semantic_mask for i in data_samples])

        losses = dict()
        losses['loss_sem_seg'] = self.loss_decode(
            seg_logits, targets, ignore_index=self.ignore_index)
        return losses

    def predict(self, inputs: SparseTensor,
                data_samples: SampleList) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (SparseTensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """
        seg_logits = self.forward(inputs)
        seg_preds = seg_logits.argmax(dim=1)

        batch_idx = inputs.C[:, -1]
        seg_pred_list = []
        for i, data_sample in enumerate(data_samples):
            seg_pred = seg_preds[batch_idx == i]
            seg_pred = seg_pred[data_sample.voxel2point_map]
            seg_pred_list.append(seg_pred)

        return seg_pred_list

    def forward(self, x: SparseTensor) -> Tensor:
        """Forward function.

        Args:
            x (SparseTensor): Features from backbone.

        Returns:
            output (Tensor): Segmentation map of shape [N, C].
                Note that output contains all points from each batch.
        """
        output = self.cls_seg(x.F)
        return output
