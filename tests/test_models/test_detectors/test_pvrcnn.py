import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestPVRCNN(unittest.TestCase):

    def test_pvrcnn(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'PointVoxelRCNN')
        DefaultScope.get_instance('test_pvrcnn', scope_name='mmdet3d')
        setup_seed(0)
        pvrcnn_cfg = get_detector_cfg(
            'pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py')
        model = MODELS.build(pvrcnn_cfg)
        num_gt_instance = 2
        packed_inputs = create_detector_inputs(num_gt_instance=num_gt_instance)

        # TODO: Support aug data test
        # aug_packed_inputs = [
        #     create_detector_inputs(num_gt_instance=num_gt_instance),
        #     create_detector_inputs(num_gt_instance=num_gt_instance + 1)
        # ]
        # test_aug_test
        # metainfo = {
        #     'pcd_scale_factor': 1,
        #     'pcd_horizontal_flip': 1,
        #     'pcd_vertical_flip': 1,
        #     'box_type_3d': LiDARInstance3DBoxes
        # }
        # for item in aug_packed_inputs:
        #     for batch_id in len(item['data_samples']):
        #         item['data_samples'][batch_id].set_metainfo(metainfo)

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

            # save the memory
            with torch.no_grad():
                losses = model.forward(**data, mode='loss')
                torch.cuda.empty_cache()
            self.assertGreater(losses['loss_rpn_cls'][0], 0)
            self.assertGreaterEqual(losses['loss_rpn_bbox'][0], 0)
            self.assertGreaterEqual(losses['loss_rpn_dir'][0], 0)
            self.assertGreater(losses['loss_semantic'], 0)
            self.assertGreaterEqual(losses['loss_bbox'], 0)
            self.assertGreaterEqual(losses['loss_cls'], 0)
            self.assertGreaterEqual(losses['loss_corner'], 0)
