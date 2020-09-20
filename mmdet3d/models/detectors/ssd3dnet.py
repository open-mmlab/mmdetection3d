from mmdet.models import DETECTORS
from .votenet import VoteNet


@DETECTORS.register_module()
class SSD3DNet(VoteNet):
    """3DSSDNet model.

    https://arxiv.org/abs/2002.10187.pdf
    """

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SSD3DNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
