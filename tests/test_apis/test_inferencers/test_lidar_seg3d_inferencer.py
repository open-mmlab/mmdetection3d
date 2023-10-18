# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest import TestCase

import mmengine
import numpy as np
import pytest
import torch
from mmengine.utils import is_list_of

from mmdet3d.apis import LidarSeg3DInferencer
from mmdet3d.structures import Det3DDataSample


class TestLiDARSeg3DInferencer(TestCase):

    def setUp(self):
        # init from alias
        self.inferencer = LidarSeg3DInferencer('pointnet2-ssg_s3dis-seg')

    def test_init(self):
        # init from metafile
        LidarSeg3DInferencer('pointnet2-ssg_s3dis-seg')
        # init from cfg
        LidarSeg3DInferencer(
            'configs/pointnet2/pointnet2_ssg_2xb16-cosine-50e_s3dis-seg.py',
            'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointnet2/pointnet2_ssg_16x2_cosine_50e_s3dis_seg-3d-13class/pointnet2_ssg_16x2_cosine_50e_s3dis_seg-3d-13class_20210514_144205-995d0119.pth'  # noqa
        )

    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            self.assertTrue(
                np.allclose(pred1['pts_semantic_mask'],
                            pred2['pts_semantic_mask']))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    @pytest.mark.skipif(
        'DISPLAY' not in os.environ, reason='requires DISPLAY device')
    def test_call(self):
        # single point cloud
        inputs = dict(points='tests/data/s3dis/points/Area_1_office_2.bin')
        torch.manual_seed(0)
        res_path = self.inferencer(inputs, return_vis=True)
        # ndarray
        pts_bytes = mmengine.fileio.get(inputs['points'])
        points = np.frombuffer(pts_bytes, dtype=np.float32)
        points = points.reshape(-1, 6)
        inputs = dict(points=points)
        torch.manual_seed(0)
        res_ndarray = self.inferencer(inputs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # multiple point clouds
        inputs = [
            dict(points='tests/data/s3dis/points/Area_1_office_2.bin'),
            dict(points='tests/data/s3dis/points/Area_1_office_2.bin')
        ]
        torch.manual_seed(0)
        res_path = self.inferencer(inputs, return_vis=True)
        # list of ndarray
        all_points = []
        for p in inputs:
            pts_bytes = mmengine.fileio.get(p['points'])
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, 6)
            all_points.append(dict(points=points))
        torch.manual_seed(0)
        res_ndarray = self.inferencer(all_points, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # point cloud dir, test different batch sizes
        pc_dir = dict(points='tests/data/s3dis/points/')
        res_bs2 = self.inferencer(pc_dir, batch_size=2, return_vis=True)
        self.assertIn('visualization', res_bs2)
        self.assertIn('predictions', res_bs2)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    @pytest.mark.skipif(
        'DISPLAY' not in os.environ, reason='requires DISPLAY device')
    def test_visualizer(self):
        inputs = dict(points='tests/data/s3dis/points/Area_1_office_2.bin')
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.inferencer(inputs, out_dir=tmp_dir)

    def test_post_processor(self):
        if not torch.cuda.is_available():
            return
        # return_datasample
        inputs = dict(points='tests/data/s3dis/points/Area_1_office_2.bin')
        res = self.inferencer(inputs, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], Det3DDataSample))

        # pred_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.inferencer(inputs, print_result=True, out_dir=tmp_dir)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', 'Area_1_office_2.json'))
            self.assertEqual(res['predictions'][0], dumped_res)
