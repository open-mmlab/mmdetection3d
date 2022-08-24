import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (_setup_seed, create_detector_inputs,
                             get_detector_cfg)


class TestImVoxelNet(unittest.TestCase):

    def test_imvoxelnet(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoxelNet')
        DefaultScope.get_instance('test_ImVoxelNet', scope_name='mmdet3d')
        _setup_seed(0)
        imvoxel_net_cfg = get_detector_cfg(
            'imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py')
        model = MODELS.build(imvoxel_net_cfg)
        num_gt_instance = 1
        data = [
            create_detector_inputs(
                with_points=False,
                with_img=True,
                img_size=(128, 128),
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

            self.assertGreaterEqual(losses['loss_cls'][0], 0)
            self.assertGreaterEqual(losses['loss_bbox'][0], 0)
            self.assertGreaterEqual(losses['loss_dir'][0], 0)
