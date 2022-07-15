# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.registry import MODELS
from .votenet import VoteNet


@MODELS.register_module()
class SSD3DNet(VoteNet):
    """3DSSDNet model.

    https://arxiv.org/abs/2002.10187.pdf
    """

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(SSD3DNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
