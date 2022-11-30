import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestFCAF3d(unittest.TestCase):

    def test_fcaf3d(self):
        try:
            import MinkowskiEngine  # noqa: F401
        except ImportError:
            return

        import mmdet3d.models
        assert hasattr(mmdet3d.models, 'MinkSingleStage3DDetector')
        DefaultScope.get_instance('test_fcaf3d', scope_name='mmdet3d')
        setup_seed(0)
        fcaf3d_net_cfg = get_detector_cfg(
            'fcaf3d/fcaf3d_2xb8_scannet-3d-18class.py')
        model = MODELS.build(fcaf3d_net_cfg)
        num_gt_instance = 3
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance,
            num_classes=1,
            points_feat_dim=6,
            gt_bboxes_dim=6)

        if torch.cuda.is_available():
            model = model.cuda()
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, False)
                torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            losses = model.forward(**data, mode='loss')

            self.assertGreater(losses['center_loss'], 0)
            self.assertGreater(losses['bbox_loss'], 0)
            self.assertGreater(losses['cls_loss'], 0)
