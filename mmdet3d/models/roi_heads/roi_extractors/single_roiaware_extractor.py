import torch
import torch.nn as nn

from mmdet3d import ops
from mmdet.models.builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class Single3DRoIAwareExtractor(nn.Module):
    """Point-wise roi-aware Extractor

    Extract Point-wise roi features.

    Args:
        roi_layer (dict): the config of roi layer
    """

    def __init__(self, roi_layer=None):
        super(Single3DRoIAwareExtractor, self).__init__()
        self.roi_layer = self.build_roi_layers(roi_layer)

    def build_roi_layers(self, layer_cfg):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = layer_cls(**cfg)
        return roi_layers

    def forward(self, feats, coordinate, batch_inds, rois):
        """Extract point-wise roi features

        Args:
            feats (FloatTensor): point-wise features with
                shape (batch, npoints, channels) for pooling
            coordinate (FloatTensor): coordinate of each point
            batch_inds (longTensor): indicate the batch of each point
            rois (FloatTensor): roi boxes with batch indices

        Returns:
            FloatTensor: pooled features
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
