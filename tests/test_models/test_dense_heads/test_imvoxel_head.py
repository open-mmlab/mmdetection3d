# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import unittest

import pytest
import torch
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d import *  # noqa
from mmdet3d.models.dense_heads import ImVoxelHead
from mmdet3d.testing import create_detector_inputs
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestImVoxelHead(TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_imvoxel_head_loss(self, _, device: str):
        """Test imvoxel head loss when truth is empty and non-empty."""
        if not is_cuda_available() and not is_musa_available():
            pytest.skip('test requires GPU and torch+cuda')

        # build head
        prior_generator = dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-3.2, -0.2, -2.28, 3.2, 6.2, 0.28]],
            rotations=[.0])
        imvoxel_head = ImVoxelHead(
            n_classes=1,
            n_levels=1,
            n_channels=32,
            n_reg_outs=7,
            pts_assign_threshold=27,
            pts_center_threshold=18,
            prior_generator=prior_generator,
            center_loss=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True),
            bbox_loss=dict(type='RotatedIoU3DLoss'),
            cls_loss=dict(type='mmdet.FocalLoss'),
        )
        imvoxel_head = imvoxel_head.to(device)

        # fake input of head
        # (x, valid_preds)
        x = [
            torch.randn(1, 32, 10, 10, 4).to(device),
            torch.ones(1, 1, 10, 10, 4).to(device)
        ]

        # fake annotation
        num_gt_instance = 1
        packed_inputs = create_detector_inputs(
            with_points=False,
            with_img=True,
            img_size=(128, 128),
            num_gt_instance=num_gt_instance,
            with_pts_semantic_mask=False,
            with_pts_instance_mask=False)
        data_samples = [
            sample.to(device) for sample in packed_inputs['data_samples']
        ]

        losses = imvoxel_head.loss(x, data_samples)
        print(losses)
        self.assertGreaterEqual(losses['center_loss'], 0)
        self.assertGreaterEqual(losses['bbox_loss'], 0)
        self.assertGreaterEqual(losses['cls_loss'], 0)
