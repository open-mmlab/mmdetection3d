import warnings
from mmcv.utils import Registry

from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS, build)
from mmseg.models.builder import SEGMENTORS

VOXEL_ENCODERS = Registry('voxel_encoder')
MIDDLE_ENCODERS = Registry('middle_encoder')
FUSION_LAYERS = Registry('fusion_layer')


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
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function warpper for building 3D detector or segmentor according to
    cfg.

    Should be deprecated in the future.
    """
    if cfg.type in ['EncoderDecoder3D']:
        return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return build(cfg, VOXEL_ENCODERS)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return build(cfg, MIDDLE_ENCODERS)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return build(cfg, FUSION_LAYERS)
