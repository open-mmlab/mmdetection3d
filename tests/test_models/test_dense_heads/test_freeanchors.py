import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestFreeAnchor(unittest.TestCase):

    def test_freeanchor(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models.dense_heads, 'FreeAnchor3DHead')
        DefaultScope.get_instance('test_freeanchor', scope_name='mmdet3d')
        setup_seed(0)
        freeanchor_cfg = get_detector_cfg(
            'free_anchor/pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor'
            '_sbn-all_8xb4-2x_nus-3d.py')
        # decrease channels to reduce cuda memory.
        freeanchor_cfg.pts_voxel_encoder.feat_channels = [1, 1]
        freeanchor_cfg.pts_middle_encoder.in_channels = 1
        freeanchor_cfg.pts_backbone.base_channels = 1
        freeanchor_cfg.pts_backbone.stem_channels = 1
        freeanchor_cfg.pts_neck.out_channels = 1
        freeanchor_cfg.pts_bbox_head.feat_channels = 1
        freeanchor_cfg.pts_bbox_head.in_channels = 1
        model = MODELS.build(freeanchor_cfg)
        num_gt_instance = 3
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance, gt_bboxes_dim=9)

        # TODO: Support aug_test
        # aug_data = [
        #     create_detector_inputs(
        #         num_gt_instance=num_gt_instance, gt_bboxes_dim=9),
        #     create_detector_inputs(
        #         num_gt_instance=num_gt_instance + 1, gt_bboxes_dim=9)
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

            self.assertGreaterEqual(losses['positive_bag_loss'], 0)
            self.assertGreaterEqual(losses['negative_bag_loss'], 0)
