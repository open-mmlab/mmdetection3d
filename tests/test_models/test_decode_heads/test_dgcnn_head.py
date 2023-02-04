# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet3d.models.decode_heads import DGCNNHead
from mmdet3d.structures import Det3DDataSample, PointData


class TestDGCNNHead(TestCase):

    def test_dgcnn_head_loss(self):
        """Tests DGCNN head loss."""

        dgcnn_head = DGCNNHead(
            fp_channels=(1024, 512),
            channels=256,
            num_classes=13,
            dropout_ratio=0.5,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            loss_decode=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            ignore_index=13)

        # DGCNN head expects dict format features
        fa_points = torch.rand(1, 4096, 1024).float()
        feat_dict = dict(fa_points=fa_points)

        # Test forward
        seg_logits = dgcnn_head.forward(feat_dict)

        self.assertEqual(seg_logits.shape, torch.Size([1, 13, 4096]))

        # When truth is non-empty then losses
        # should be nonzero for random inputs
        pts_semantic_mask = torch.randint(0, 13, (4096, )).long()
        gt_pts_seg = PointData(pts_semantic_mask=pts_semantic_mask)

        datasample = Det3DDataSample()
        datasample.gt_pts_seg = gt_pts_seg

        gt_losses = dgcnn_head.loss_by_feat(seg_logits, [datasample])

        gt_sem_seg_loss = gt_losses['loss_sem_seg'].item()

        self.assertGreater(gt_sem_seg_loss, 0,
                           'semantic seg loss should be positive')
