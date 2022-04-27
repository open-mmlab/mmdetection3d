# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector


@DETECTORS.register_module()
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
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(SingleStage3DDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_dummy(self, points):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(points)
        try:
            sample_mod = self.train_cfg.sample_mod
            outs = self.bbox_head(x, sample_mod)
        except AttributeError:
            outs = self.bbox_head(x)
        return outs

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img_meta)
            for pts, img_meta in zip(points, img_metas)
        ]
