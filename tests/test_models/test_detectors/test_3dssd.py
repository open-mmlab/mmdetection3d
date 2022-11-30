import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class Test3DSSD(unittest.TestCase):

    def test_3dssd(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'SSD3DNet')
        DefaultScope.get_instance('test_ssd3d', scope_name='mmdet3d')
        setup_seed(0)
        voxel_net_cfg = get_detector_cfg('3dssd/3dssd_4xb4_kitti-3d-car.py')
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

            self.assertGreater(losses['centerness_loss'], 0)
