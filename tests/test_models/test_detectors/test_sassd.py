import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestSDSSD(unittest.TestCase):

    def test_3dssd(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'SASSD')
        DefaultScope.get_instance('test_sassd', scope_name='mmdet3d')
        setup_seed(0)
        voxel_net_cfg = get_detector_cfg(
            'sassd/sassd_8xb6-80e_kitti-3d-3class.py')
        model = MODELS.build(voxel_net_cfg)
        num_gt_instance = 3
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance, num_classes=1)

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            losses = model.forward(**data, mode='loss')
            self.assertGreaterEqual(losses['loss_dir'][0], 0)
            self.assertGreaterEqual(losses['loss_bbox'][0], 0)
            self.assertGreaterEqual(losses['loss_cls'][0], 0)
            self.assertGreater(losses['aux_loss_cls'][0], 0)
            self.assertGreater(losses['aux_loss_reg'][0], 0)
