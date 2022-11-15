# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads import BaseRoIHead

from mmdet3d.registry import MODELS, TASK_UTILS


class Base3DRoIHead(BaseRoIHead):
    """Base class for 3d RoIHeads."""

    def __init__(self,
                 bbox_head=None,
                 bbox_roi_extractor=None,
                 mask_head=None,
                 mask_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Base3DRoIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_head=mask_head,
            mask_roi_extractor=mask_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    TASK_UTILS.build(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)

    def init_mask_head(self):
        """Initialize mask head, skip since ``PartAggregationROIHead`` does not
        have one."""
        pass
