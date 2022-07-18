# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.core import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.registry import MODELS
from .single_stage_mono3d import SingleStageMono3DDetector


@MODELS.register_module()
class FCOSMono3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of FCOS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of FCOS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
