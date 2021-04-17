from mmdet.models.builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector


@DETECTORS.register_module()
class FCOSMono3D(SingleStageMono3DDetector):
    """Implementation of FCOS3D. The technical report will be released soon.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>` # noqa
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSMono3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
