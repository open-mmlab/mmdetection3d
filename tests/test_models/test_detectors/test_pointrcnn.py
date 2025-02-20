import unittest

import torch
from mmengine import DefaultScope
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestPointRCNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_pointrcnn(self, _, device: str):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'PointRCNN')
        DefaultScope.get_instance('test_pointrcnn', scope_name='mmdet3d')
        setup_seed(0)
        pointrcnn_cfg = get_detector_cfg(
            'point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py')
        model = MODELS.build(pointrcnn_cfg)
        num_gt_instance = 2
        packed_inputs = create_detector_inputs(
            num_points=10101, num_gt_instance=num_gt_instance)

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                if is_musa_available():
                    torch.mush.empty_cache()
                else:
                    torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(**data, mode='loss')
                torch.cuda.empty_cache()
            self.assertGreaterEqual(losses['rpn_bbox_loss'], 0)
            self.assertGreaterEqual(losses['rpn_semantic_loss'], 0)
            self.assertGreaterEqual(losses['loss_cls'], 0)
            self.assertGreaterEqual(losses['loss_bbox'], 0)
            self.assertGreaterEqual(losses['loss_corner'], 0)
