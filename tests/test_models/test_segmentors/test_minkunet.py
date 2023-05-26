# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import pytest
import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestMinkUNet(unittest.TestCase):

    def test_minkunet(self):
        try:
            import torchsparse  # noqa
        except ImportError:
            pytest.skip('test requires Torchsparse installation')

        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'MinkUNet')
        DefaultScope.get_instance('test_minkunet', scope_name='mmdet3d')
        setup_seed(0)
        model_cfg = get_detector_cfg('_base_/models/minkunet.py')
        model = MODELS.build(model_cfg)
        num_gt_instance = 3
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance,
            num_classes=19,
            with_pts_semantic_mask=True)

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('pts_semantic_mask', results[0].pred_pts_seg)

            losses = model.forward(**data, mode='loss')

            self.assertGreater(losses['loss_sem_seg'], 0)
