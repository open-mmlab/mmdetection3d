# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch

from mmdet3d.registry import MODELS
from .two_stage import TwoStage3DDetector


@MODELS.register_module()
class PointRCNN(TwoStage3DDetector):
    r"""PointRCNN detector.

    Please refer to the `PointRCNN <https://arxiv.org/abs/1812.04244>`_

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        rpn_head (dict, optional): Config of RPN head. Defaults to None.
        roi_head (dict, optional): Config of ROI head. Defaults to None.
        train_cfg (dict, optional): Train configs. Defaults to None.
        test_cfg (dict, optional): Test configs. Defaults to None.
        pretrained (str, optional): Model pretrained path. Defaults to None.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None) -> None:
        super(PointRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

    def extract_feat(self, batch_inputs_dict: Dict) -> Dict:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image of each sample.

        Returns:
            dict: Features from the backbone+neck and raw points.
        """
        points = torch.stack(batch_inputs_dict['points'])
        x = self.backbone(points)

        if self.with_neck:
            x = self.neck(x)
        return dict(
            fp_features=x['fp_features'].clone(),
            fp_points=x['fp_xyz'].clone(),
            raw_points=points)
