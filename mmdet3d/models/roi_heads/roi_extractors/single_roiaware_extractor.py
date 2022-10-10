# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmcv import ops
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class Single3DRoIAwareExtractor(BaseModule):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict, optional): The config of roi layer.
    """

    def __init__(self,
                 roi_layer: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super(Single3DRoIAwareExtractor, self).__init__(init_cfg=init_cfg)
        self.roi_layer = self.build_roi_layers(roi_layer)

    def build_roi_layers(self, layer_cfg: dict) -> nn.Module:
        """Build roi layers using `layer_cfg`"""
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = layer_cls(**cfg)
        return roi_layers

    def forward(self, feats: Tensor, coordinate: Tensor, batch_inds: Tensor,
                rois: Tensor) -> Tensor:
        """Extract point-wise roi features.

        Args:
            feats (torch.FloatTensor): Point-wise features with
                shape (batch, npoints, channels) for pooling.
            coordinate (torch.FloatTensor): Coordinate of each point.
            batch_inds (torch.LongTensor): Indicate the batch of each point.
            rois (torch.FloatTensor): Roi boxes with batch indices.

        Returns:
            torch.FloatTensor: Pooled features
        """
        pooled_roi_feats = []
        for batch_idx in range(int(batch_inds.max()) + 1):
            roi_inds = (rois[..., 0].int() == batch_idx)
            coors_inds = (batch_inds.int() == batch_idx)
            pooled_roi_feat = self.roi_layer(rois[..., 1:][roi_inds],
                                             coordinate[coors_inds],
                                             feats[coors_inds])
            pooled_roi_feats.append(pooled_roi_feat)
        pooled_roi_feats = torch.cat(pooled_roi_feats, 0)
        return pooled_roi_feats
