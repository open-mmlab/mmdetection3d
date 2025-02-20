import unittest

import torch
from mmengine import DefaultScope
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestImvoteNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_imvotenet_only_img(self, _, device: str):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoteNet')
        DefaultScope.get_instance('test_imvotenet_img', scope_name='mmdet3d')
        setup_seed(0)
        votenet_net_cfg = get_detector_cfg(
            'imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py')
        model = MODELS.build(votenet_net_cfg)

        packed_inputs = create_detector_inputs(
            with_points=False, with_img=True, img_size=128)

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes', results[0].pred_instances)
            self.assertIn('scores', results[0].pred_instances)
            self.assertIn('labels', results[0].pred_instances)

            # save the memory
            with torch.no_grad():
                if is_musa_available():
                    torch.musa.empty_cache()
                else:
                    torch.cuda.empty_cache()
                losses = model.forward(**data, mode='loss')

            self.assertGreater(sum(losses['loss_rpn_cls']), 0)

            self.assertGreater(losses['loss_cls'], 0)
            self.assertGreater(losses['loss_bbox'], 0)

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_imvotenet(self, _, device: str):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoteNet')
        DefaultScope.get_instance('test_imvotenet', scope_name='mmdet3d')
        setup_seed(0)
        votenet_net_cfg = get_detector_cfg(
            'imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py')
        model = MODELS.build(votenet_net_cfg)

        packed_inputs = create_detector_inputs(
            with_points=True,
            with_img=True,
            img_size=128,
            bboxes_3d_type='depth')

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(**data, mode='loss')

            self.assertGreater(losses['vote_loss'], 0)
            self.assertGreater(losses['objectness_loss'], 0)
            self.assertGreater(losses['semantic_loss'], 0)
