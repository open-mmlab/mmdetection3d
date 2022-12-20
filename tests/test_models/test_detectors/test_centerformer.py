import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestCenterFormer(unittest.TestCase):

    def test_centerformer(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'CenterFormer')

        setup_seed(0)
        DefaultScope.get_instance('test_centerformer', scope_name='mmdet3d')
        centerformer_net_cfg = get_detector_cfg(
            'centerformer/centerformer_voxel01_second_4xb4-cyclic-20e_waymoD5-3d-3class.py'  # noqa
        )
        # centerformer_net_cfg.backbone.out_channels = [16, 16]
        # centerformer_net_cfg.neck.in_channels = [16, 16]
        model = MODELS.build(centerformer_net_cfg)
        num_gt_instance = 50
        packed_inputs = create_detector_inputs(
            with_img=False, num_gt_instance=num_gt_instance, points_feat_dim=5)

        for sample_id in range(len(packed_inputs['data_samples'])):
            det_sample = packed_inputs['data_samples'][sample_id]
            num_instances = len(det_sample.gt_instances_3d.bboxes_3d)
            bbox_3d_class = det_sample.gt_instances_3d.bboxes_3d.__class__
            det_sample.gt_instances_3d.bboxes_3d = bbox_3d_class(
                torch.rand(num_instances, 7), box_dim=7)

        if torch.cuda.is_available():

            model = model.cuda()
            # test simple_test

            data = model.data_preprocessor(packed_inputs, True)
            with torch.no_grad():
                torch.cuda.empty_cache()
                losses = model.forward(**data, mode='loss')
            assert losses['task0.loss_heatmap'] >= 0
            assert losses['task0.loss_bbox'] >= 0
            assert losses['task0.loss_corner'] >= 0
            assert losses['task0.loss_iou'] >= 0

            with torch.no_grad():
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0])
            self.assertIn('scores_3d', results[0])
            self.assertIn('labels_3d', results[0])
