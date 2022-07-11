# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from mmcv.ops import Voxelization
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from .two_stage import TwoStage3DDetector


@MODELS.register_module()
class PartA2(TwoStage3DDetector):
    r"""Part-A2 detector.

    Please refer to the `paper <https://arxiv.org/abs/1907.03670>`_
    """

    def __init__(self,
                 voxel_layer: dict,
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
        self.voxel_layer = Voxelization(**voxel_layer)
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
        points = batch_inputs_dict['points']
        voxel_dict = self.voxelize(points)
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

    @torch.no_grad()
    def voxelize(self, points: List[torch.Tensor]) -> Dict:
        """Apply hard voxelization to points."""
        voxels, coors, num_points, voxel_centers = [], [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                        self.voxel_layer.point_cloud_range[0:3])
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxel_centers.append(res_voxel_centers)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        voxel_centers = torch.cat(voxel_centers, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        voxel_dict = dict(
            voxels=voxels,
            num_points=num_points,
            coors=coors_batch,
            voxel_centers=voxel_centers)
        return voxel_dict
