import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestSSN(unittest.TestCase):

    def test_ssn(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models.dense_heads, 'ShapeAwareHead')
        DefaultScope.get_instance('test_ssn', scope_name='mmdet3d')
        setup_seed(0)
        ssn_cfg = get_detector_cfg(
            'ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_nus-3d.py')
        ssn_cfg.pts_voxel_encoder.feat_channels = [1, 1]
        ssn_cfg.pts_middle_encoder.in_channels = 1
        ssn_cfg.pts_backbone.in_channels = 1
        ssn_cfg.pts_backbone.out_channels = [1, 1, 1]
        ssn_cfg.pts_neck.in_channels = [1, 1, 1]
        ssn_cfg.pts_neck.out_channels = [1, 1, 1]
        ssn_cfg.pts_bbox_head.in_channels = 3
        ssn_cfg.pts_bbox_head.feat_channels = 1
        model = MODELS.build(ssn_cfg)
        num_gt_instance = 50
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance, gt_bboxes_dim=9)

        # TODO: Support aug_test
        # aug_data = [
        #     create_detector_inputs(
        #         num_gt_instance=num_gt_instance, gt_bboxes_dim=9),
        #     create_detector_inputs(
        #         num_gt_instance=num_gt_instance + 1, gt_bboxes_dim=9)
        # ]
        # test_aug_test
        # metainfo = {
        #     'pcd_scale_factor': 1,
        #     'pcd_horizontal_flip': 1,
        #     'pcd_vertical_flip': 1,
        #     'box_type_3d': LiDARInstance3DBoxes
        # }
        # for item in aug_data:
        #     item['data_sample'].set_metainfo(metainfo)

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

            losses = model.forward(**data, mode='loss')

            self.assertGreaterEqual(losses['loss_cls'][0], 0)
            self.assertGreaterEqual(losses['loss_bbox'][0], 0)
            self.assertGreaterEqual(losses['loss_dir'][0], 0)
