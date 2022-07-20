import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestPartA2(unittest.TestCase):

    def test_parta2(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'PartA2')
        DefaultScope.get_instance('test_parta2', scope_name='mmdet3d')
        _setup_seed(0)
        parta2_cfg = _get_detector_cfg(
            'parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py')
        model = MODELS.build(parta2_cfg)
        num_gt_instance = 2
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
            # save the memory

            with torch.no_grad():
                losses = model.forward(batch_inputs, data_samples, mode='loss')
                torch.cuda.empty_cache()
            self.assertGreater(losses['loss_rpn_cls'][0], 0)
            self.assertGreater(losses['loss_rpn_bbox'][0], 0)
            self.assertGreater(losses['loss_seg'], 0)
            self.assertGreater(losses['loss_part'], 0)
            self.assertGreater(losses['loss_cls'], 0)
