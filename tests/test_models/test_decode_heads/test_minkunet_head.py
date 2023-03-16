# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn.functional as F

from mmdet3d.models.decode_heads import MinkUNetHead
from mmdet3d.structures import Det3DDataSample, PointData


class TestMinkUNetHead(TestCase):

    def test_minkunet_head_loss(self):
        """Tests PAConv head loss."""

        try:
            import torchsparse
        except ImportError:
            pytest.skip('test requires Torchsparse installation')
        if torch.cuda.is_available():
            minkunet_head = MinkUNetHead(channels=4, num_classes=19)

            minkunet_head.cuda()
            coordinates, features = [], []
            for i in range(2):
                c = torch.randint(0, 10, (100, 3)).int()
                c = F.pad(c, (0, 1), mode='constant', value=i)
                coordinates.append(c)
                f = torch.rand(100, 4)
                features.append(f)
            features = torch.cat(features, dim=0).cuda()
            coordinates = torch.cat(coordinates, dim=0).cuda()
            x = torchsparse.SparseTensor(feats=features, coords=coordinates)

            # Test forward
            seg_logits = minkunet_head.forward(x)

            self.assertEqual(seg_logits.shape, torch.Size([200, 19]))

            # When truth is non-empty then losses
            # should be nonzero for random inputs
            voxel_semantic_mask = torch.randint(0, 19, (100, )).long().cuda()
            gt_pts_seg = PointData(voxel_semantic_mask=voxel_semantic_mask)

            datasample = Det3DDataSample()
            datasample.gt_pts_seg = gt_pts_seg

            gt_losses = minkunet_head.loss(x, [datasample, datasample], {})

            gt_sem_seg_loss = gt_losses['loss_sem_seg'].item()

            self.assertGreater(gt_sem_seg_loss, 0,
                               'semantic seg loss should be positive')
