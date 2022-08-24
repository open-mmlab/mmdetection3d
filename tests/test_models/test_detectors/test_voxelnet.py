# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.testing import (_setup_seed, create_detector_inputs,
                             get_detector_cfg)


class TestVoxelNet(unittest.TestCase):

    def test_voxelnet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'VoxelNet')
        DefaultScope.get_instance('test_voxelnet', scope_name='mmdet3d')
        _setup_seed(0)
        pointpillars_cfg = get_detector_cfg(
            'pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py')
        model = MODELS.build(pointpillars_cfg)
        num_gt_instance = 2
        data = [create_detector_inputs(num_gt_instance=num_gt_instance)]
        aug_data = [
            create_detector_inputs(num_gt_instance=num_gt_instance),
            create_detector_inputs(num_gt_instance=num_gt_instance + 1)
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
            self.assertGreater(losses['loss_dir'][0], 0)
            self.assertGreater(losses['loss_bbox'][0], 0)
            self.assertGreater(losses['loss_cls'][0], 0)
