# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import unittest

import pytest
import torch
from mmcv.ops import SparseConvTensor
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d.models.decode_heads import Cylinder3DHead
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestCylinder3DHead(TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_cylinder3d_head_loss(self, _, device: str):
        """Tests Cylinder3D head loss."""
        if not is_cuda_available() and not is_musa_available():
            pytest.skip('test requires GPU and torch+cuda')
        cylinder3d_head = Cylinder3DHead(
            channels=128,
            num_classes=20,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.0, reduction='none'),
        ).to(device)

        voxel_feats = torch.rand(50, 128).to(device)
        coorx = torch.randint(0, 480, (50, 1)).int().to(device)
        coory = torch.randint(0, 360, (50, 1)).int().to(device)
        coorz = torch.randint(0, 32, (50, 1)).int().to(device)
        coorbatch0 = torch.zeros(50, 1).int().to(device)
        coors = torch.cat([coorbatch0, coorx, coory, coorz], dim=1)
        grid_size = [480, 360, 32]
        batch_size = 1

        sparse_voxels = SparseConvTensor(voxel_feats, coors, grid_size,
                                         batch_size)
        # Test forward
        seg_logits = cylinder3d_head.forward(sparse_voxels)

        self.assertEqual(seg_logits.features.shape, torch.Size([50, 20]))

        # When truth is non-empty then losses
        # should be nonzero for random inputs
        voxel_semantic_mask = torch.randint(0, 20, (50, )).long().to(device)
        gt_pts_seg = PointData(voxel_semantic_mask=voxel_semantic_mask)

        datasample = Det3DDataSample()
        datasample.gt_pts_seg = gt_pts_seg

        losses = cylinder3d_head.loss_by_feat(seg_logits, [datasample])

        loss_ce = losses['loss_ce'].item()
        loss_lovasz = losses['loss_lovasz'].item()

        self.assertGreater(loss_ce, 0, 'ce loss should be positive')
        self.assertGreater(loss_lovasz, 0, 'lovasz loss should be positive')

        batch_inputs_dict = dict(voxels=dict(voxel_coors=coors))
        datasample.point2voxel_map = torch.randint(0, 50, (100, )).int().to(device)
        point_logits = cylinder3d_head.predict(sparse_voxels,
                                               batch_inputs_dict, [datasample])
        assert point_logits[0].shape == torch.Size([100, 20])
