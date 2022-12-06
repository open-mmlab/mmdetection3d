import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestMVXNet(unittest.TestCase):

    def test_mvxnet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'DynamicMVXFasterRCNN')

        setup_seed(0)
        DefaultScope.get_instance('test_mvxnet', scope_name='mmdet3d')
        mvx_net_cfg = get_detector_cfg(
            'mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py'  # noqa
        )
        model = MODELS.build(mvx_net_cfg)
        num_gt_instance = 1
        packed_inputs = create_detector_inputs(
            with_img=False, num_gt_instance=num_gt_instance, points_feat_dim=4)

        if torch.cuda.is_available():

            model = model.cuda()
            # test simple_test
            data = model.data_preprocessor(packed_inputs, True)
            # save the memory when do the unitest
            with torch.no_grad():
                torch.cuda.empty_cache()
                losses = model.forward(**data, mode='loss')
            assert losses['loss_cls'][0] >= 0
            assert losses['loss_bbox'][0] >= 0
            assert losses['loss_dir'][0] >= 0

            with torch.no_grad():
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)
        # TODO test_aug_test
