from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS, build)
from .registry import FUSION_LAYERS, MIDDLE_ENCODERS, VOXEL_ENCODERS


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss function."""
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return build(cfg, VOXEL_ENCODERS)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return build(cfg, MIDDLE_ENCODERS)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return build(cfg, FUSION_LAYERS)
