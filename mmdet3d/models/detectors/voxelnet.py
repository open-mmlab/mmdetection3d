# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
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
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x
