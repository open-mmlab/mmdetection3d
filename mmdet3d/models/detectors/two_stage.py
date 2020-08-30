from mmdet.models import DETECTORS, TwoStageDetector
from .base import Base3DDetector


@DETECTORS.register_module()
class TwoStage3DDetector(Base3DDetector, TwoStageDetector):
    """Base class of two-stage 3D detector.

    It inherits original ``:class:TwoStageDetector`` and
    ``:class:Base3DDetector``. This class could serve as a base class
    for all two-stage 3D detectors.
    """

    def __init__(self, **kwargs):
        super(TwoStage3DDetector, self).__init__(**kwargs)
