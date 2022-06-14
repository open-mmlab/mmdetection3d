# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models import TwoStageDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector


@DETECTORS.register_module()
class TwoStage3DDetector(Base3DDetector, TwoStageDetector):
    """Base class of two-stage 3D detector.

    It inherits original ``:class:TwoStageDetector`` and
    ``:class:Base3DDetector``. This class could serve as a base class for all
    two-stage 3D detectors.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
