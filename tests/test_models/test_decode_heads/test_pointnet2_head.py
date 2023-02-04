# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet3d.models.decode_heads import PointNet2Head
from mmdet3d.structures import Det3DDataSample, PointData


class TestPointNet2Head(TestCase):

    def test_paconv_head_loss(self):
        """Tests PAConv head loss."""

        if torch.cuda.is_available():
            pointnet2_head = PointNet2Head(
                fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
                             (128, 128, 128, 128)),
                channels=128,
                num_classes=20,
                dropout_ratio=0.5,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                act_cfg=dict(type='ReLU'),
                loss_decode=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight=None,
                    loss_weight=1.0),
                ignore_index=20)

            pointnet2_head.cuda()

            # DGCNN head expects dict format features
            sa_xyz = [
                torch.rand(1, 4096, 3).float().cuda(),
                torch.rand(1, 1024, 3).float().cuda(),
                torch.rand(1, 256, 3).float().cuda(),
                torch.rand(1, 64, 3).float().cuda(),
                torch.rand(1, 16, 3).float().cuda(),
            ]
            sa_features = [
                torch.rand(1, 6, 4096).float().cuda(),
                torch.rand(1, 64, 1024).float().cuda(),
                torch.rand(1, 128, 256).float().cuda(),
                torch.rand(1, 256, 64).float().cuda(),
                torch.rand(1, 512, 16).float().cuda(),
            ]
            feat_dict = dict(sa_xyz=sa_xyz, sa_features=sa_features)

            # Test forward
            seg_logits = pointnet2_head.forward(feat_dict)

            self.assertEqual(seg_logits.shape, torch.Size([1, 20, 4096]))

            # When truth is non-empty then losses
            # should be nonzero for random inputs
            pts_semantic_mask = torch.randint(0, 20, (4096, )).long().cuda()
            gt_pts_seg = PointData(pts_semantic_mask=pts_semantic_mask)

            datasample = Det3DDataSample()
            datasample.gt_pts_seg = gt_pts_seg

            gt_losses = pointnet2_head.loss_by_feat(seg_logits, [datasample])

            gt_sem_seg_loss = gt_losses['loss_sem_seg'].item()

            self.assertGreater(gt_sem_seg_loss, 0,
                               'semantic seg loss should be positive')
