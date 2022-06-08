# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import Det3DDataSample
from mmdet3d.registry import MODELS
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer: dict,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None) -> None:
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, points: List[torch.Tensor]) -> list:
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points: List[torch.Tensor]) -> tuple:
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self, batch_inputs_dict: Dict[list, torch.Tensor],
                      batch_data_samples: List[Det3DDataSample],
                      **kwargs) -> dict:
        """
        Args:
            batch_inputs_dict (dict): The model input dict. It should contain
                ``points`` and ``img`` keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (list[:obj:`Det3DDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance_3d` or `gt_panoptic_seg_3d` or `gt_sem_seg_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        x = self.extract_feat(batch_inputs_dict['points'])
        losses = self.bbox_head.forward_train(x, batch_data_samples, **kwargs)
        return losses

    def simple_test(self,
                    batch_inputs_dict: Dict[list, torch.Tensor],
                    batch_input_metas: List[dict],
                    rescale: bool = False) -> list:
        """Test function without test-time augmentation.

        Args:
            batch_inputs_dict (dict): The model input dict. It should contain
                ``points`` and ``img`` keys.

                    - points (list[torch.Tensor]): Point cloud of single
                        sample.
                    - imgs (torch.Tensor, optional): Image of single sample.

            batch_input_metas (list[dict]): List of input information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the \
                inputs. Each Det3DDataSample usually contain \
                'pred_instances_3d'. And the ``pred_instances_3d`` usually \
                contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instances, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                    contains a tensor with shape (num_instances, 7).
        """
        x = self.extract_feat(batch_inputs_dict['points'])
        bboxes_list = self.bbox_head.simple_test(
            x, batch_input_metas, rescale=rescale)

        # connvert to Det3DDataSample
        results_list = self.postprocess_result(bboxes_list)
        return results_list

    def aug_test(self,
                 aug_batch_inputs_dict: Dict[list, torch.Tensor],
                 aug_batch_input_metas: List[dict],
                 rescale: bool = False) -> list:
        """Test function with augmentaiton."""
        # TODO Refactor this after mmdet update
        feats = self.extract_feats(aug_batch_inputs_dict)
        aug_bboxes = self.bbox_head.aug_test(
            feats, aug_batch_input_metas, rescale=rescale)
        return aug_bboxes
