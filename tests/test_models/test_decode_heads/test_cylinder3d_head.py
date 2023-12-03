# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from mmcv.ops import SparseConvTensor

from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.models.decode_heads import Cylinder3DHead
from mmdet3d.structures import Det3DDataSample, PointData


class TestCylinder3DHead(TestCase):

    def test_cylinder3d_head_loss(self):
        """Tests Cylinder3D head loss."""
        if not torch.cuda.is_available():
            pytest.skip('test requires GPU and torch+cuda')
        cylinder3d_head = Cylinder3DHead(
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.0, reduction='none'),
            conv_seg_kernel_size=3,
            ignore_index=19).cuda()

        pts_feats = torch.rand(100, 128).cuda()
        coorx = torch.randint(0, 480, (100, 1)).int().cuda()
        coory = torch.randint(0, 360, (100, 1)).int().cuda()
        coorz = torch.randint(0, 32, (100, 1)).int().cuda()
        coorbatch0 = torch.zeros(100, 1).int().cuda()
        coors = torch.cat([coorbatch0, coorx, coory, coorz], dim=1)
        voxel_feats, voxel_coors, point2voxel_map = dynamic_scatter_3d(
            pts_feats, coors)
        grid_size = [480, 360, 32]
        batch_size = 1

        sparse_voxels = SparseConvTensor(voxel_feats, voxel_coors, grid_size,
                                         batch_size)
        voxel_dict = dict(
            voxel_feats=sparse_voxels,
            voxel_coors=voxel_coors,
            coors=coors,
            point2voxel_maps=[point2voxel_map])
        # Test forward
        voxel_dict = cylinder3d_head.forward(voxel_dict)

        self.assertEqual(voxel_dict['logits'].shape,
                         torch.Size([voxel_coors.shape[0], 20]))

        # When truth is non-empty then losses
        # should be nonzero for random inputs
        pts_semantic_mask = torch.randint(0, 20, (100, )).long().cuda()
        gt_pts_seg = PointData(pts_semantic_mask=pts_semantic_mask)

        datasample = Det3DDataSample()
        datasample.gt_pts_seg = gt_pts_seg

        losses = cylinder3d_head.loss_by_feat(voxel_dict, [datasample])

        loss_ce = losses['loss_ce'].item()
        loss_lovasz = losses['loss_lovasz'].item()

        self.assertGreater(loss_ce, 0, 'ce loss should be positive')
        self.assertGreater(loss_lovasz, 0, 'lovasz loss should be positive')

        point_logits = cylinder3d_head.predict(voxel_dict, [datasample], None)
        assert point_logits[0].shape == torch.Size([100, 20])
