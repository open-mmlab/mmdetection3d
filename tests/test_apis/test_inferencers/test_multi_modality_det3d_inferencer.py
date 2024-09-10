# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.utils import is_list_of

from mmdet3d.apis import MultiModalityDet3DInferencer
from mmdet3d.structures import Det3DDataSample


class TestMultiModalityDet3DInferencer(TestCase):

    def setUp(self):
        # init from alias
        self.inferencer = MultiModalityDet3DInferencer('mvxnet_kitti-3class')

    def test_init(self):
        # init from metafile
        MultiModalityDet3DInferencer('mvxnet_kitti-3class')
        # init from cfg
        MultiModalityDet3DInferencer(
            'configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py',  # noqa
            weights=  # noqa
            'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth'  # noqa
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
        infos_path = 'demo/data/kitti/000008.pkl'
        points_path = 'demo/data/kitti/000008.bin'
        img_path = 'demo/data/kitti/000008.png'
        # single img & point cloud
        inputs = dict(points=points_path, img=img_path, infos=infos_path)
        res_path = self.inferencer(inputs, return_vis=True)

        # ndarray
        pts_bytes = mmengine.fileio.get(inputs['points'])
        points = np.frombuffer(pts_bytes, dtype=np.float32)
        points = points.reshape(-1, 4)
        points = points[:, :4]
        img = mmcv.imread(inputs['img'])
        inputs = dict(points=points, img=img, infos=infos_path)
        res_ndarray = self.inferencer(inputs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # multiple imgs & point clouds
        inputs = [
            dict(points=points_path, img=img_path, infos=infos_path),
            dict(points=points_path, img=img_path, infos=infos_path)
        ]
        res_path = self.inferencer(inputs, return_vis=True)
        # list of ndarray
        all_inputs = []
        for p in inputs:
            pts_bytes = mmengine.fileio.get(p['points'])
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, 4)
            img = mmcv.imread(p['img'])
            all_inputs.append(dict(points=points, img=img, infos=infos_path))

        res_ndarray = self.inferencer(all_inputs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

    def test_visualize(self):
        if not torch.cuda.is_available():
            return
        inputs = dict(
            points='demo/data/kitti/000008.bin',
            img='demo/data/kitti/000008.png',
            infos='demo/data/kitti/000008.pkl'),
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.inferencer(inputs, out_dir=tmp_dir)
            # TODO: For results of LiDAR-based detection, the saved image only
            # exists when show=True.
            # self.assertTrue(osp.exists(osp.join(tmp_dir, '000000.png')))

    def test_postprocess(self):
        if not torch.cuda.is_available():
            return
        # return_datasample
        infos_path = 'demo/data/kitti/000008.pkl'
        points_path = 'demo/data/kitti/000008.bin'
        img_path = 'demo/data/kitti/000008.png'
        # single img & point cloud
        inputs = dict(points=points_path, img=img_path, infos=infos_path)
        res = self.inferencer(inputs, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], Det3DDataSample))

        # pred_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = dict(points=points_path, img=img_path, infos=infos_path)
            res = self.inferencer(inputs, print_result=True, out_dir=tmp_dir)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', '000008.json'))
            self.assertEqual(res['predictions'][0], dumped_res)
