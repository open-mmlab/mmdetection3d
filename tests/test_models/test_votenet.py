import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.registry import MODELS
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestVotenet(unittest.TestCase):

    def test_voxel_net(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'VoteNet')
        DefaultScope.get_instance('test_vote_net', scope_name='mmdet3d')
        _setup_seed(0)
        voxel_net_cfg = _get_detector_cfg(
            'votenet/votenet_16x8_sunrgbd-3d-10class.py')
        model = MODELS.build(voxel_net_cfg)
        num_gt_instance = 50
        data = [_create_detector_inputs(num_gt_instance=num_gt_instance)]
        aug_data = [
            _create_detector_inputs(num_gt_instance=num_gt_instance),
            _create_detector_inputs(num_gt_instance=num_gt_instance + 1)
        ]
        # test_aug_test
        metainfo = {
            'pcd_scale_factor': 1,
            'pcd_horizontal_flip': 1,
            'pcd_vertical_flip': 1,
            'box_type_3d': LiDARInstance3DBoxes
        }
        for item in aug_data:
            item['data_sample'].set_metainfo(metainfo)
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
            batch_inputs, data_samples = model.data_preprocessor(
                aug_data, True)
            aug_results = model.forward(
                batch_inputs, data_samples, mode='predict')

            self.assertIn('bboxes_3d', aug_results[0].pred_instances_3d)
            self.assertIn('scores_3d', aug_results[0].pred_instances_3d)
            self.assertIn('labels_3d', aug_results[0].pred_instances_3d)

            losses = model.forward(batch_inputs, data_samples, mode='loss')

            self.assertGreater(losses['vote_loss'], 0)
            self.assertGreater(losses['objectness_loss'], 0)
            self.assertGreater(losses['semantic_loss'], 0)
            self.assertGreater(losses['dir_res_loss'], 0)
            self.assertGreater(losses['size_class_loss'], 0)
            self.assertGreater(losses['size_res_loss'], 0)
            self.assertGreater(losses['size_res_loss'], 0)

        # TODO test_aug_test
