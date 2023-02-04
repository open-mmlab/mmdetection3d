# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmdet3d.registry import MODELS
from .two_stage import TwoStage3DDetector


@MODELS.register_module()
class PartA2(TwoStage3DDetector):
    r"""Part-A2 detector.

    Please refer to the `paper <https://arxiv.org/abs/1907.03670>`_
    """

    def __init__(self,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 backbone: dict,
                 neck: dict = None,
                 rpn_head: dict = None,
                 roi_head: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None,
                 data_preprocessor: Optional[dict] = None):
        super(PartA2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, batch_inputs_dict: Dict) -> Dict:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                         batch_size)
        x = self.backbone(feats_dict['spatial_features'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict.update({'neck_feats': neck_feats})
        feats_dict['voxels_dict'] = voxel_dict
        return feats_dict
