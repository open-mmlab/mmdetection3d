import unittest

import torch
from mmengine import DefaultScope
from mmengine.device import is_musa_available, is_cuda_available
from parameterized import parameterized

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)
from mmdet3d.testing.model_utils import AVAILABLE_DEVICES


class TestImVoxelNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (is_cuda_available() or is_musa_available()):
            raise unittest.SkipTest("requires CUDA or MUSA support")

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_imvoxelnet_kitti(self, _, device: str):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoxelNet')
        DefaultScope.get_instance(
            'test_imvoxelnet_kitti', scope_name='mmdet3d')
        setup_seed(0)
        imvoxel_net_cfg = get_detector_cfg(
            'imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py')
        model = MODELS.build(imvoxel_net_cfg)
        num_gt_instance = 1
        packed_inputs = create_detector_inputs(
            with_points=False,
            with_img=True,
            img_size=(128, 128),
            num_gt_instance=num_gt_instance,
            with_pts_semantic_mask=False,
            with_pts_instance_mask=False)

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                if is_musa_available():
                    torch.musa.empty_cache()
                else:
                    torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(**data, mode='loss')

            self.assertGreaterEqual(losses['loss_cls'][0], 0)
            self.assertGreaterEqual(losses['loss_bbox'][0], 0)
            self.assertGreaterEqual(losses['loss_dir'][0], 0)

    @parameterized.expand(AVAILABLE_DEVICES)
    def test_imvoxelnet_sunrgbd(self, _, device: str):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ImVoxelNet')
        DefaultScope.get_instance(
            'test_imvoxelnet_sunrgbd', scope_name='mmdet3d')
        setup_seed(0)
        imvoxel_net_cfg = get_detector_cfg(
            'imvoxelnet/imvoxelnet_2xb4_sunrgbd-3d-10class.py')
        model = MODELS.build(imvoxel_net_cfg)
        num_gt_instance = 1
        packed_inputs = create_detector_inputs(
            with_points=False,
            with_img=True,
            img_size=(128, 128),
            num_gt_instance=num_gt_instance,
            with_pts_semantic_mask=False,
            with_pts_instance_mask=False)

        if is_cuda_available() or is_musa_available():
            model = model.to(device)
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                if is_musa_available():
                    torch.musa.empty_cache()
                else:
                    torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
            self.assertEqual(len(results), 1)
            self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            self.assertIn('scores_3d', results[0].pred_instances_3d)
            self.assertIn('labels_3d', results[0].pred_instances_3d)

            # save the memory
            with torch.no_grad():
                losses = model.forward(**data, mode='loss')

            self.assertGreaterEqual(losses['center_loss'], 0)
            self.assertGreaterEqual(losses['bbox_loss'], 0)
            self.assertGreaterEqual(losses['cls_loss'], 0)
