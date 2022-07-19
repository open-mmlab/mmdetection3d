import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from tests.utils.model_utils import (_create_detector_inputs,
                                     _get_detector_cfg, _setup_seed)


class TestImVoxelNet(unittest.TestCase):

    def test_h3dnet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoxelNet')
        DefaultScope.get_instance('test_ImVoxelNet', scope_name='mmdet3d')
        _setup_seed(0)
        imvoxel_net_cfg = _get_detector_cfg(
            'imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py')
        model = MODELS.build(imvoxel_net_cfg)
        num_gt_instance = 1
        data = [
            _create_detector_inputs(
                with_points=False,
                with_img=True,
                img_size=(384, 1280),
                num_gt_instance=num_gt_instance,
                with_pts_semantic_mask=False,
                with_pts_instance_mask=False)
        ]

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                batch_inputs, data_samples = model.data_preprocessor(
                    data, True)
                results = model.forward(
                    batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(results), len(data))
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(batch_inputs, data_samples, mode='loss')

            self.assertGreater(losses['loss_cls'], 0)
            self.assertGreater(losses['loss_bbox'], 0)
            self.assertGreater(losses['loss_dir'], 0)
