# Copyright (c) OpenMMLab. All rights reserved.
import unittest
import pytest

import torch
from mmengine import DefaultScope
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestCylinder3D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_cylinder3d(self, _, device):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'Cylinder3D')
        DefaultScope.get_instance('test_cylinder3d', scope_name='mmdet3d')
        setup_seed(0)
        cylinder3d_cfg = get_detector_cfg(
            'cylinder3d/cylinder3d_4xb4-3x_semantickitti.py')
        cylinder3d_cfg.decode_head['ignore_index'] = 1
        model = MODELS.build(cylinder3d_cfg)
        num_gt_instance = 3
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance,
            num_classes=1,
            with_pts_semantic_mask=True)

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                if is_musa_available():
                    torch.musa.empty_cache()
                else:
                    torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('pts_semantic_mask', results[0].pred_pts_seg)

            losses = model.forward(**data, mode='loss')

            self.assertGreater(losses['decode.loss_ce'], 0)
            self.assertGreater(losses['decode.loss_lovasz'], 0)
