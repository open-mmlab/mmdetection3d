# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor, nn

from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType


@MODELS.register_module()
class RangeImageHead(Base3DDecodeHead):
    """RangeImage decoder head.

    Args:
        loss_ce (dict or :obj:`ConfigDict`): Config of CrossEntropy loss.
            Defaults to dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0).
        loss_lovasz (dict or :obj:`ConfigDict`, optional): Config of Lovasz
            loss. Defaults to None.
        lpss_boundary (dict or :obj:`ConfigDict`, optional): Config of boundary
            loss. Defaults to None.
        indices (int): The indice of features to use. Defaults to 0.
    """

    def __init__(self,
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_lovasz: OptConfigType = None,
                 loss_boundary: OptConfigType = None,
                 indices: int = 0,
                 **kwargs) -> None:
        super(RangeImageHead, self).__init__(**kwargs)

        self.loss_ce = MODELS.build(loss_ce)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None
        if loss_boundary is not None:
            self.loss_boundary = MODELS.build(loss_boundary)
        else:
            self.loss_boundary = None

        self.indices = indices

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return nn.Conv2d(channels, num_classes, kernel_size=kernel_size)

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward function."""
        seg_logit = self.cls_seg(feats[self.indices])
        return seg_logit

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.semantic_seg
            for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logit: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (Tensor): Predicted  logits.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_label = seg_label.squeeze(dim=1)
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_lovasz:
            loss['loss_lovasz'] = self.loss_lovasz(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_boundary:
            loss['loss_boundary'] = self.loss_boundary(seg_logit, seg_label)
        return loss

    def predict(self, inputs: Tuple[Tensor], batch_input_metas: List[dict],
                test_cfg: ConfigType) -> torch.Tensor:
        """Forward function for testing.

        Args:
            inputs (Tuple[Tensor]): Features from backbone.
            batch_input_metas (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. We use `point2voxel_map` in this function.
            test_cfg (dict or :obj:`ConfigDict`): The testing config.

        Returns:
            List[Tensor]: List of point-wise segmentation labels.
        """
        seg_logits = self.forward(inputs)
        seg_labels = seg_logits.argmax(dim=1)
        device = seg_logits.device
        use_knn = test_cfg.get('use_knn', False)
        if use_knn:
            from .utils import KNN
            post_module = KNN(
                test_cfg=test_cfg,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index)

        seg_label_list = []
        for i in range(len(batch_input_metas)):
            input_metas = batch_input_metas[i]
            proj_x = torch.tensor(
                input_metas['proj_x'], dtype=torch.int64, device=device)
            proj_y = torch.tensor(
                input_metas['proj_y'], dtype=torch.int64, device=device)
            proj_range = torch.tensor(
                input_metas['proj_range'], dtype=torch.float32, device=device)
            unproj_range = torch.tensor(
                input_metas['unproj_range'],
                dtype=torch.float32,
                device=device)

            if use_knn:
                seg_label_list.append(
                    post_module(proj_range, unproj_range, seg_labels[i],
                                proj_x, proj_y))
            else:
                seg_label_list.append(seg_labels[i, proj_y, proj_x])

        return seg_label_list
