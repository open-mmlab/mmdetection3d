# FocalFormer3D plugin for mmdetection3d 1.4 / mmcv 2.2.
# The module layout mirrors NVlabs/FocalFormer3D
# (projects/mmdet3d_plugin/...)

# FocalFormer3D components need CUDA, which login nodes do not have.
#
# Keep this try/except. On a CPU-only node the import raises and these become
# None, and the FAILURE MODE IS CONFUSING: MODELS.build later reports
# "FocalFormer3D is not in the mmdet3d::model registry", which reads like a
# config bug but only means there was no GPU. Run registry checks in a GPU job.
try:
    from .models.dense_heads.focal_decoder import FocalDecoder
    from .models.detectors.focalformer3d import FocalFormer3D
    from .models.necks.focal_encoder import FocalEncoder
except (AssertionError, ImportError):
    FocalEncoder = None
    FocalDecoder = None
    FocalFormer3D = None

from .core.bbox.assigners.hungarian_assigner import (HeuristicAssigner3D,
                                                     HungarianAssigner3D)
from .core.bbox.coders.transfusion_bbox_coder import TransFusionBBoxCoder
from .core.hook.fading import Fading
from .datasets.pipelines import FFImageAug3D, FFLoadMultiViewImage

__all__ = [
    'FocalEncoder',
    'FocalDecoder',
    'FocalFormer3D',
    'HungarianAssigner3D',
    'HeuristicAssigner3D',
    'TransFusionBBoxCoder',
    'Fading',
    'FFLoadMultiViewImage',
    'FFImageAug3D',
]
