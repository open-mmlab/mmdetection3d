# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmengine
import numpy as np
import torch
from mmengine.utils import is_list_of

from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.structures import Det3DDataSample


class TestLidarDet3DInferencer(TestCase):

    def setUp(self):
        # init from alias
        self.inferencer = LidarDet3DInferencer('pointpillars_kitti-3class')

    def test_init(self):
        # init from metafile
        LidarDet3DInferencer('pointpillars_waymod5-3class')
        # init from cfg
        LidarDet3DInferencer(
            'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',  # noqa
            weights=  # noqa
            'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'  # noqa
        )

    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            if 'bboxes_3d' in pred1:
                self.assertTrue(
                    np.allclose(pred1['bboxes_3d'], pred2['bboxes_3d'], 0.1))
            if 'scores_3d' in pred1:
                self.assertTrue(
                    np.allclose(pred1['scores_3d'], pred2['scores_3d'], 0.1))
            if 'labels_3d' in pred1:
                self.assertTrue(
                    np.allclose(pred1['labels_3d'], pred2['labels_3d']))

    def test_call(self):
        if not torch.cuda.is_available():
            return
        # single point cloud
        inputs = dict(points='tests/data/kitti/training/velodyne/000000.bin')
        res_path = self.inferencer(inputs, return_vis=True)
        # ndarray
        pts_bytes = mmengine.fileio.get(inputs['points'])
        points = np.frombuffer(pts_bytes, dtype=np.float32)
        points = points.reshape(-1, 4)
        points = points[:, :4]
        inputs = dict(points=points)
        res_ndarray = self.inferencer(inputs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # multiple point clouds
        inputs = [
            dict(points='tests/data/kitti/training/velodyne/000000.bin'),
            dict(points='tests/data/kitti/training/velodyne/000000.bin')
        ]
        res_path = self.inferencer(inputs, return_vis=True)
        # list of ndarray
        all_points = []
        for p in inputs:
            pts_bytes = mmengine.fileio.get(p['points'])
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, 4)
            all_points.append(dict(points=points))
        res_ndarray = self.inferencer(all_points, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # point cloud dir, test different batch sizes
        pc_dir = dict(points='tests/data/kitti/training/velodyne/')
        res_bs2 = self.inferencer(pc_dir, batch_size=2, return_vis=True)
        self.assertIn('visualization', res_bs2)
        self.assertIn('predictions', res_bs2)

    def test_visualize(self):
        if not torch.cuda.is_available():
            return
        inputs = dict(points='tests/data/kitti/training/velodyne/000000.bin'),
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.inferencer(inputs, out_dir=tmp_dir)
            # TODO: For LiDAR-based detection, the saved image only exists when
            # show=True.
            # self.assertTrue(osp.exists(osp.join(tmp_dir, '000000.png')))

    def test_postprocess(self):
        if not torch.cuda.is_available():
            return
        # return_datasample
        inputs = dict(points='tests/data/kitti/training/velodyne/000000.bin')
        res = self.inferencer(inputs, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], Det3DDataSample))

        # pred_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.inferencer(inputs, print_result=True, out_dir=tmp_dir)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', '000000.json'))
            self.assertEqual(res['predictions'][0], dumped_res)
