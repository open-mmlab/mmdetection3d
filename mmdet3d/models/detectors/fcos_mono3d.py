from mmdet.models.builder import DETECTORS
from .single_stage_mono3d import SingleStageDetectorMono3D


@DETECTORS.register_module()
class FCOSMono3D(SingleStageDetectorMono3D):
    """Implementation of FCOS3D. Technical report will be released soon.

    Currently you can refer to our entry on the
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
