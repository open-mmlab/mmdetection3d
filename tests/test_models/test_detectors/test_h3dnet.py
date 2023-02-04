import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestH3D(unittest.TestCase):

    def test_h3dnet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'H3DNet')
        DefaultScope.get_instance('test_H3DNet', scope_name='mmdet3d')
        setup_seed(0)
        voxel_net_cfg = get_detector_cfg('h3dnet/h3dnet_8xb3_scannet-seg.py')
        model = MODELS.build(voxel_net_cfg)
        num_gt_instance = 5
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance,
            points_feat_dim=4,
            bboxes_3d_type='depth',
            with_pts_semantic_mask=True,
            with_pts_instance_mask=True)

        if torch.cuda.is_available():
            model = model.cuda()
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
            self.assertGreater(losses['center_loss'], 0)
