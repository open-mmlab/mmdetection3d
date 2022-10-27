# Copyright (c) OpenMMLab. All rights reserved.
from .foreground_segmentation_head import ForegroundSegmentationHead
from .pointwise_semantic_head import PointwiseSemanticHead
from .primitive_head import PrimitiveHead

__all__ = [
    'PointwiseSemanticHead', 'PrimitiveHead', 'ForegroundSegmentationHead'
]
