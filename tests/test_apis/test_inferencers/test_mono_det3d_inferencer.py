# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import mmengine
import numpy as np
from mmengine.utils import is_list_of
from parameterized import parameterized

from mmdet3d.apis import MonoDet3DInferencer
from mmdet3d.structures import Det3DDataSample


class TestMonoDet3DInferencer(TestCase):

    def test_init(self):
        # init from metafile
        MonoDet3DInferencer('pgd_kitti')
        # init from cfg
        MonoDet3DInferencer(
            'configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py',
            'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/'
            'pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/'
            'pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_'
            '20211022_102608-8a97533b.pth')

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

    @parameterized.expand(['pgd_kitti'])
    def test_call(self, model):
        # single img
        img_path = 'demo/data/kitti/000008.png'
        infos_path = 'demo/data/kitti/000008.pkl'
        inferencer = MonoDet3DInferencer(model)
        inputs = dict(img=img_path, infos=infos_path)
        res_path = inferencer(inputs, return_vis=True)
        # ndarray
        img = mmcv.imread(img_path)
        inputs = dict(img=img, infos=infos_path)
        res_ndarray = inferencer(inputs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # multiple images
        inputs = [
            dict(
                img='demo/data/kitti/000008.png',
                infos='demo/data/kitti/000008.pkl'),
            dict(
                img='demo/data/kitti/000008.png',
                infos='demo/data/kitti/000008.pkl')
        ]
        res_path = inferencer(inputs, return_vis=True)
        # list of ndarray
        imgs = [mmcv.imread(p['img']) for p in inputs]
        inputs = [
            dict(img=imgs[0], infos='demo/data/kitti/000008.pkl'),
            dict(img=imgs[1], infos='demo/data/kitti/000008.pkl')
        ]
        res_ndarray = inferencer(inputs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

    @parameterized.expand(['pgd_kitti'])
    def test_visualize(self, model):
        inputs = dict(
            img='demo/data/kitti/000008.png',
            infos='demo/data/kitti/000008.pkl')
        inferencer = MonoDet3DInferencer(model)
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            inferencer(inputs, out_dir=tmp_dir)
            self.assertTrue(
                osp.exists(osp.join(tmp_dir, 'vis_camera/CAM2/000008.png')))

    @parameterized.expand(['pgd_kitti'])
    def test_postprocess(self, model):
        # return_datasample
        img_path = 'demo/data/kitti/000008.png'
        infos_path = 'demo/data/kitti/000008.pkl'
        inputs = dict(img=img_path, infos=infos_path)
        inferencer = MonoDet3DInferencer(model)
        res = inferencer(inputs, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], Det3DDataSample))

        # pred_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = dict(img=img_path, infos=infos_path)
            res = inferencer(inputs, print_result=True, out_dir=tmp_dir)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', '000008.json'))
            self.assertEqual(res['predictions'][0], dumped_res)
