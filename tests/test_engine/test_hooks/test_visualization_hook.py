# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.engine.hooks import Det3DVisualizationHook
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        Det3DLocalVisualizer.get_instance('visualizer')

        pred_instances_3d = InstanceData()
        pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(
                [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900, -1.5808]]))
        pred_instances_3d.labels_3d = torch.tensor([0])
        pred_instances_3d.scores_3d = torch.tensor([0.8])

        pred_det3d_data_sample = Det3DDataSample()
        pred_det3d_data_sample.set_metainfo({
            'num_pts_feats':
            4,
            'lidar2img':
            np.array([[
                6.02943734e+02, -7.07913286e+02, -1.22748427e+01,
                -1.70942724e+02
            ],
                      [
                          1.76777261e+02, 8.80879902e+00, -7.07936120e+02,
                          -1.02568636e+02
                      ],
                      [
                          9.99984860e-01, -1.52826717e-03, -5.29071223e-03,
                          -3.27567990e-01
                      ],
                      [
                          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                          1.00000000e+00
                      ]]),
            'img_path':
            osp.join(
                osp.dirname(__file__),
                '../../data/kitti/training/image_2/000000.png'),
            'lidar_path':
            osp.join(
                osp.dirname(__file__),
                '../../data/kitti/training/velodyne_reduced/000000.bin')
        })
        pred_det3d_data_sample.pred_instances_3d = pred_instances_3d
        self.outputs = [pred_det3d_data_sample] * 2

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = Det3DVisualizationHook()
        hook.after_val_iter(runner, 1, {}, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = Det3DVisualizationHook(draw=True)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertEqual(hook._test_index, 2)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = Det3DVisualizationHook(draw=False, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{test_out_dir}'))

        hook = Det3DVisualizationHook(draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')
