from typing import Dict, List, Optional

import torch
from torch import Tensor

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class TransFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(TransFusionDetector, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', True)
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        super(TransFusionDetector, self).init_weights()

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
