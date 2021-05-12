from mmdet.models.builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector


@DETECTORS.register_module()
class KMMono3D(SingleStageMono3DDetector):
    r"""KMMono3D <https://arxiv.org/abs/2009.00764v1>`_ for monocular 3D object detection.

    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KMMono3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
