# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine import DefaultScope
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestVoxelNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_voxelnet(self, _, device: str):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'VoxelNet')
        DefaultScope.get_instance('test_voxelnet', scope_name='mmdet3d')
        setup_seed(0)
        pointpillars_cfg = get_detector_cfg(
            'pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py')
        model = MODELS.build(pointpillars_cfg)
        num_gt_instance = 2
        packed_inputs = create_detector_inputs(num_gt_instance=num_gt_instance)

        # TODO: Support aug_test
        # aug_data = [
        #     create_detector_inputs(num_gt_instance=num_gt_instance),
        #     create_detector_inputs(num_gt_instance=num_gt_instance + 1)
        # ]
        # # test_aug_test
        # metainfo = {
        #     'pcd_scale_factor': 1,
        #     'pcd_horizontal_flip': 1,
        #     'pcd_vertical_flip': 1,
        #     'box_type_3d': LiDARInstance3DBoxes
        # }
        # for item in aug_data:
        #     item['data_sample'].set_metainfo(metainfo)

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                if is_musa_available():
                    torch.mush.empty_cache()
                else:
                    torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # TODO: Support aug_test
            # batch_inputs, data_samples = model.data_preprocessor(
            #     aug_data, True)
            # aug_results = model.forward(
            #     batch_inputs, data_samples, mode='predict')
            # self.assertEqual(len(results), len(data))
            # self.assertIn('bboxes_3d', aug_results[0].pred_instances_3d)
            # self.assertIn('scores_3d', aug_results[0].pred_instances_3d)
            # self.assertIn('labels_3d', aug_results[0].pred_instances_3d)
            # self.assertIn('bboxes_3d', aug_results[1].pred_instances_3d)
            # self.assertIn('scores_3d', aug_results[1].pred_instances_3d)
            # self.assertIn('labels_3d', aug_results[1].pred_instances_3d)

            # save the memory

            with torch.no_grad():
                losses = model.forward(**data, mode='loss')
                if is_musa_available():
                    torch.mush.empty_cache()
                else:
                    torch.cuda.empty_cache()
            self.assertGreaterEqual(losses['loss_dir'][0], 0)
            self.assertGreaterEqual(losses['loss_bbox'][0], 0)
            self.assertGreaterEqual(losses['loss_cls'][0], 0)
