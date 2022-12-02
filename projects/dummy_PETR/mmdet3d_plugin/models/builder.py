# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
from mmdet.models.builder import DETECTORS as MMDET_DETECTORS
from mmdet.models.builder import HEADS as MMDET_HEADS
from mmdet.models.builder import LOSSES as MMDET_LOSSES
from mmdet.models.builder import NECKS as MMDET_NECKS
from mmdet.models.builder import ROI_EXTRACTORS as MMDET_ROI_EXTRACTORS
from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
from mmseg.models.builder import LOSSES as MMSEG_LOSSES

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    if cfg['type'] in BACKBONES._module_dict.keys():
        return BACKBONES.build(cfg)
    else:
        return MMDET_BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    if cfg['type'] in NECKS._module_dict.keys():
        return NECKS.build(cfg)
    else:
        return MMDET_NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    if cfg['type'] in ROI_EXTRACTORS._module_dict.keys():
        return ROI_EXTRACTORS.build(cfg)
    else:
        return MMDET_ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    if cfg['type'] in SHARED_HEADS._module_dict.keys():
        return SHARED_HEADS.build(cfg)
    else:
        return MMDET_SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    if cfg['type'] in HEADS._module_dict.keys():
        return HEADS.build(cfg)
    else:
        return MMDET_HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    if cfg['type'] in LOSSES._module_dict.keys():
        return LOSSES.build(cfg)
    elif cfg['type'] in MMDET_LOSSES._module_dict.keys():
        return MMDET_LOSSES.build(cfg)
    else:
        return MMSEG_LOSSES.build(cfg)


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
    if cfg['type'] in DETECTORS._module_dict.keys():
        return DETECTORS.build(
            cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    else:
        return MMDET_DETECTORS.build(
            cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


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
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function wrapper for building 3D detector or segmentor according to
    cfg.

    Should be deprecated in the future.
    """
    if cfg.type in ['EncoderDecoder3D']:
        return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)
