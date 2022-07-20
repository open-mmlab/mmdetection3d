import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestFreeAnchor(unittest.TestCase):

    def test_freeanchor(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models.dense_heads, 'FreeAnchor3DHead')
        DefaultScope.get_instance('test_freeanchor', scope_name='mmdet3d')
        _setup_seed(0)
        freeanchor_cfg = _get_detector_cfg(
            'free_anchor/hv_pointpillars_fpn_sbn-all_free-'
            'anchor_4x8_2x_nus-3d.py')
        model = MODELS.build(freeanchor_cfg)
        num_gt_instance = 3
        data = [
            _create_detector_inputs(
                num_gt_instance=num_gt_instance, gt_bboxes_dim=9)
        ]
        aug_data = [
            _create_detector_inputs(
                num_gt_instance=num_gt_instance, gt_bboxes_dim=9),
            _create_detector_inputs(
                num_gt_instance=num_gt_instance + 1, gt_bboxes_dim=9)
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
                torch.cuda.empty_cache()
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
            self.assertEqual(len(results), len(data))
            self.assertIn('bboxes_3d', aug_results[0].pred_instances_3d)
            self.assertIn('scores_3d', aug_results[0].pred_instances_3d)
            self.assertIn('labels_3d', aug_results[0].pred_instances_3d)
            self.assertIn('bboxes_3d', aug_results[1].pred_instances_3d)
            self.assertIn('scores_3d', aug_results[1].pred_instances_3d)
            self.assertIn('labels_3d', aug_results[1].pred_instances_3d)

            losses = model.forward(batch_inputs, data_samples, mode='loss')

            self.assertGreater(losses['positive_bag_loss'], 0)
            self.assertGreater(losses['negative_bag_loss'], 0)
