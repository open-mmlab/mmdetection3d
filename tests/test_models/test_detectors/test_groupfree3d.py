import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (_setup_seed, create_detector_inputs,
                             get_detector_cfg)


class TestGroupfree3d(unittest.TestCase):

    def test_groupfree3d(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'GroupFree3DNet')
        DefaultScope.get_instance('test_groupfree3d', scope_name='mmdet3d')
        _setup_seed(0)
        voxel_net_cfg = get_detector_cfg(
            'groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py')
        model = MODELS.build(voxel_net_cfg)
        num_gt_instance = 5
        data = [
            create_detector_inputs(
                num_gt_instance=num_gt_instance,
                points_feat_dim=3,
                with_pts_semantic_mask=True,
                with_pts_instance_mask=True)
        ]

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

            # save the memory
            with torch.no_grad():
                losses = model.forward(batch_inputs, data_samples, mode='loss')

            self.assertGreater(losses['sampling_objectness_loss'], 0)
            self.assertGreater(losses['proposal.objectness_loss'], 0)
            self.assertGreater(losses['s0.objectness_loss'], 0)
            self.assertGreater(losses['s1.size_res_loss'], 0)
            self.assertGreater(losses['s4.size_class_loss'], 0)
