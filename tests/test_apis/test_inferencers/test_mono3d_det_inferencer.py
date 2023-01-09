# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import mmengine
import numpy as np
from mmengine.utils import is_list_of

from mmdet3d.apis import MonoDet3DInferencer
from mmdet3d.structures import Det3DDataSample


class TestMonoDet3DInferencer(TestCase):

    def test_init(self):
        # init from metafile
        MonoDet3DInferencer('pgd-kitti')
        # init from cfg
        MonoDet3DInferencer(
            'configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py')

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

    def test_call(self, model):
        # single img
        img_path = 'tests/data/kitti/training/image_2/000007.png'
        inferencer = MonoDet3DInferencer(model)
        res_path = inferencer(img_path, return_vis=True)
        # ndarray
        img = mmcv.imread(img_path)
        res_ndarray = inferencer(img, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # multiple images
        img_paths = [
            'tests/data/kitti/training/image_2/000007.png',
            'tests/data/kitti/training/image_2/000000.png'
        ]
        res_path = inferencer(img_paths, return_vis=True)
        # list of ndarray
        imgs = [mmcv.imread(p) for p in img_paths]
        res_ndarray = inferencer(imgs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # img dir, test different batch sizes
        img_dir = 'tests/data/kitti/training/image_2/'
        res_bs1 = inferencer(img_dir, batch_size=1, return_vis=True)
        res_bs3 = inferencer(img_dir, batch_size=2, return_vis=True)
        self.assert_predictions_equal(res_bs1['predictions'],
                                      res_bs3['predictions'])
        if model == 'pgd-kitti':
            # There is a jitter operation when the mask is drawn,
            # so it cannot be asserted.
            for res_bs1_vis, res_bs3_vis in zip(res_bs1['visualization'],
                                                res_bs3['visualization']):
                self.assertTrue(np.allclose(res_bs1_vis, res_bs3_vis))

    def test_visualize(self, model):
        img_paths = [
            'tests/data/kitti/training/image_2/000007.png',
            'tests/data/kitti/training/image_2/000000.png'
        ]
        inferencer = MonoDet3DInferencer(model)
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            inferencer(img_paths, img_out_dir=tmp_dir)
            for img_dir in ['000007.png', '000000.png']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, img_dir)))

    def test_postprocess(self, model):
        # return_datasample
        img_path = 'tests/data/kitti/training/image_2/000007.png'
        inferencer = MonoDet3DInferencer(model)
        res = inferencer(img_path, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], Det3DDataSample))

        # pred_out_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_out_file = osp.join(tmp_dir, 'tmp.json')
            res = inferencer(
                img_path, print_result=True, pred_out_file=pred_out_file)
            dumped_res = mmengine.load(pred_out_file)
            self.assert_predictions_equal(res['predictions'],
                                          dumped_res['predictions'])
