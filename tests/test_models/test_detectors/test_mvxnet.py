import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestMVXNet(unittest.TestCase):

    def test_mvxnet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'DynamicMVXFasterRCNN')

        _setup_seed(0)
        DefaultScope.get_instance('test_mvxnet', scope_name='mmdet3d')
        mvx_net_cfg = _get_detector_cfg(
            'mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py'  # noqa
        )
        model = MODELS.build(mvx_net_cfg)
        num_gt_instance = 50
        data = [
            _create_detector_inputs(
                with_img=False,
                num_gt_instance=num_gt_instance,
                points_feat_dim=4)
        ]

        if torch.cuda.is_available():

            model = model.cuda()
            # test simple_test
            batch_inputs, data_samples = model.data_preprocessor(data, True)
            # save the memory when do the unitest
            with torch.no_grad():
                losses = model.forward(batch_inputs, data_samples, mode='loss')
            assert losses['loss_cls'][0] >= 0
            assert losses['loss_bbox'][0] >= 0
            assert losses['loss_dir'][0] >= 0

            with torch.no_grad():
                results = model.forward(
                    batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(results), len(data))
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)
        # TODO test_aug_test
