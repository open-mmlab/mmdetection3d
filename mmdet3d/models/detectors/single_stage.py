# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from mmdet3d.registry import MODELS
from .base import Base3DDetector


@MODELS.register_module()
class SingleStage3DDetector(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        pretrained (str, optional): Path of pretrained models.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 preprocess_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None) -> None:
        super(SingleStage3DDetector, self).__init__(
            preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_dummy(self, batch_inputs: dict) -> tuple:
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(batch_inputs['points'])
        try:
            sample_mod = self.train_cfg.sample_mod
            outs = self.bbox_head(x, sample_mod)
        except AttributeError:
            outs = self.bbox_head(x)
        return outs

    def extract_feat(self, points: List[torch.Tensor]) -> list:
        """Directly extract features from the backbone+neck.

        Args:
            points (List[torch.Tensor]): Input points.
        """
        x = self.backbone(points[0])
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, batch_inputs_dict: dict) -> list:
        """Extract features of multiple samples."""
        return [
            self.extract_feat([points])
            for points in batch_inputs_dict['points']
        ]
