import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestCenterPoint(unittest.TestCase):

    def test_center_point(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'CenterPoint')

        _setup_seed(0)
        DefaultScope.get_instance('test_center_point', scope_name='mmdet3d')
        centerpoint_net_cfg = _get_detector_cfg(
            'centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py'  # noqa
        )
        model = MODELS.build(centerpoint_net_cfg)
        num_gt_instance = 50
        data = [
            _create_detector_inputs(
                with_img=True,
                num_gt_instance=num_gt_instance,
                points_feat_dim=5)
        ]
        for sample_id in range(len(data)):
            det_sample = data[sample_id]['data_sample']
            num_instances = len(det_sample.gt_instances_3d.bboxes_3d)
            bbox_3d_class = det_sample.gt_instances_3d.bboxes_3d.__class__
            det_sample.gt_instances_3d.bboxes_3d = bbox_3d_class(
                torch.rand(num_instances, 9), box_dim=9)

        if torch.cuda.is_available():

            model = model.cuda()
            # test simple_test

            batch_inputs, data_samples = model.data_preprocessor(data, True)

            losses = model.forward(batch_inputs, data_samples, mode='loss')
            assert losses['task0.loss_heatmap'] >= 0
            assert losses['task0.loss_bbox'] >= 0
            assert losses['task1.loss_heatmap'] >= 0
            assert losses['task1.loss_bbox'] >= 0
            assert losses['task2.loss_heatmap'] >= 0
            assert losses['task2.loss_bbox'] >= 0
            assert losses['task3.loss_heatmap'] >= 0
            assert losses['task3.loss_bbox'] >= 0
            assert losses['task3.loss_bbox'] >= 0
            assert losses['task4.loss_bbox'] >= 0
            assert losses['task5.loss_heatmap'] >= 0
            assert losses['task5.loss_bbox'] >= 0

            with torch.no_grad():
                results = model.forward(
                    batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(results), len(data))
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)
        # TODO test_aug_test
