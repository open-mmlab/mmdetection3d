import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestImvoteNet(unittest.TestCase):

    def test_imvotenet_only_img(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoteNet')
        DefaultScope.get_instance('test_imvotenet_img', scope_name='mmdet3d')
        _setup_seed(0)
        votenet_net_cfg = _get_detector_cfg(
            'imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py')
        model = MODELS.build(votenet_net_cfg)

        data = [
            _create_detector_inputs(
                with_points=False, with_img=True, img_size=128)
        ]

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                batch_inputs, data_samples = model.data_preprocessor(
                    data, True)
                results = model.forward(
                    batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(results), len(data))
            self.assertIn('bboxes', results[0].pred_instances)
            self.assertIn('scores', results[0].pred_instances)
            self.assertIn('labels', results[0].pred_instances)

            # save the memory
            with torch.no_grad():
                torch.cuda.empty_cache()
                losses = model.forward(batch_inputs, data_samples, mode='loss')

            self.assertGreater(sum(losses['loss_rpn_cls']), 0)

            self.assertGreater(losses['loss_cls'], 0)
            self.assertGreater(losses['loss_bbox'], 0)

    def test_imvotenet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoteNet')
        DefaultScope.get_instance('test_imvotenet', scope_name='mmdet3d')
        _setup_seed(0)
        votenet_net_cfg = _get_detector_cfg(
            'imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py')
        model = MODELS.build(votenet_net_cfg)

        data = [
            _create_detector_inputs(
                with_points=True,
                with_img=True,
                img_size=128,
                bboxes_3d_type='depth')
        ]

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                batch_inputs, data_samples = model.data_preprocessor(
                    data, True)
                results = model.forward(
                    batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(results), len(data))
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(batch_inputs, data_samples, mode='loss')

            self.assertGreater(losses['vote_loss'], 0)
            self.assertGreater(losses['objectness_loss'], 0)
            self.assertGreater(losses['semantic_loss'], 0)
