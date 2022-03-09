# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor
from .single_roiaware_extractor import Single3DRoIAwareExtractor
from .single_roipoint_extractor import Single3DRoIPointExtractor

__all__ = [
    'SingleRoIExtractor', 'Single3DRoIAwareExtractor',
    'Single3DRoIPointExtractor'
]
