# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import unittest

import pytest
import torch
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d import *  # noqa
from mmdet3d.models.dense_heads import FCAF3DHead
from mmdet3d.testing import create_detector_inputs
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestFCAF3DHead(TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_fcaf3d_head_loss(self, _, device: str):
        """Test fcaf3d head loss when truth is empty and non-empty."""
        if not is_cuda_available() and not is_musa_available():
            pytest.skip('test requires GPU and torch+cuda')

        try:
            import MinkowskiEngine as ME
        except ImportError:
            pytest.skip('test requires MinkowskiEngine installation')

        # build head
        fcaf3d_head = FCAF3DHead(
            in_channels=(64, 128, 256, 512),
            out_channels=128,
            voxel_size=1.,
            pts_prune_threshold=1000,
            pts_assign_threshold=27,
            pts_center_threshold=18,
            num_classes=18,
            num_reg_outs=6,
            test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01),
            center_loss=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True),
            bbox_loss=dict(type='AxisAlignedIoULoss'),
            cls_loss=dict(type='mmdet.FocalLoss'),
        )
        fcaf3d_head = fcaf3d_head.to(device)

        # fake input of head
        coordinates, features = [torch.randn(500, 3).to(device) * 100
                                 ], [torch.randn(500, 3).to(device)]
        tensor_coordinates, tensor_features = ME.utils.sparse_collate(
            coordinates, features)
        x = ME.SparseTensor(
            features=tensor_features, coordinates=tensor_coordinates)
        # backbone
        conv1 = ME.MinkowskiConvolution(
            3, 64, kernel_size=3, stride=2, dimension=3).to(device)
        conv2 = ME.MinkowskiConvolution(
            64, 128, kernel_size=3, stride=2, dimension=3).to(device)
        conv3 = ME.MinkowskiConvolution(
            128, 256, kernel_size=3, stride=2, dimension=3).to(device)
        conv4 = ME.MinkowskiConvolution(
            256, 512, kernel_size=3, stride=2, dimension=3).to(device)

        # backbone outputs of 4 levels
        x1 = conv1(x)
        x2 = conv2(x1)
        x3 = conv3(x2)
        x4 = conv4(x3)
        x = (x1, x2, x3, x4)

        # fake annotation
        packed_inputs = create_detector_inputs(
            with_points=False,
            with_img=False,
            num_gt_instance=3,
            num_classes=1,
            points_feat_dim=6,
            gt_bboxes_dim=6)
        data_samples = [
            sample.to(device) for sample in packed_inputs['data_samples']
        ]

        gt_losses = fcaf3d_head.loss(x, data_samples)
        print(gt_losses)
        self.assertGreaterEqual(gt_losses['cls_loss'], 0,
                                'cls loss should be non-zero')
        self.assertGreaterEqual(gt_losses['bbox_loss'], 0,
                                'box loss should be non-zero')
        self.assertGreaterEqual(gt_losses['center_loss'], 0,
                                'dir loss should be none-zero')
