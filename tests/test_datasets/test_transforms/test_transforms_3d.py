# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets import GlobalAlignment, RandomFlip3D
from mmdet3d.datasets.transforms import GlobalRotScaleTrans
from mmdet3d.testing import create_data_info_after_loading


class TestGlobalRotScaleTrans(unittest.TestCase):

    def test_globle_rotation_scale_trans(self):
        rot_trans = GlobalRotScaleTrans(
            rot_range=[-0.78, 0.78], scale_ratio_range=[1, 1])
        scale_trans = GlobalRotScaleTrans(
            rot_range=[0, 0], scale_ratio_range=[0.95, 1.05])

        ori_data_info = create_data_info_after_loading()

        data_info = copy.deepcopy(ori_data_info)
        rot_data_info = rot_trans(data_info)
        self.assertIn('pcd_rotation', rot_data_info)
        self.assertIn('pcd_rotation_angle', rot_data_info)
        self.assertIn('pcd_scale_factor', rot_data_info)
        self.assertEqual(rot_data_info['pcd_scale_factor'], 1)
        self.assertIs(-0.79 < rot_data_info['pcd_rotation_angle'] < 0.79, True)

        # assert the rot angle should in rot_range
        before_rot_gt_bbox_3d = ori_data_info['gt_bboxes_3d']
        after_rot_gt_bbox_3d = rot_data_info['gt_bboxes_3d']
        assert (after_rot_gt_bbox_3d.tensor[:, -1] -
                before_rot_gt_bbox_3d.tensor[:, -1]).abs().max() < 0.79

        data_info = copy.deepcopy(ori_data_info)
        scale_data_info = scale_trans(data_info)
        # assert the rot angle should in rot_range
        before_scale_gt_bbox_3d = ori_data_info['gt_bboxes_3d'].tensor
        after_scale_gt_bbox_3d = scale_data_info['gt_bboxes_3d'].tensor
        before_scale_points = ori_data_info['points'].tensor
        after_scale_points = scale_data_info['points'].tensor
        self.assertEqual(scale_data_info['pcd_rotation_angle'], 0)
        # assert  scale_factor range
        assert (0.94 < (after_scale_points / before_scale_points)).all()
        assert (1.06 >
                (after_scale_gt_bbox_3d / before_scale_gt_bbox_3d)).all()


class TestRandomFlip3D(unittest.TestCase):

    def test_random_flip3d(self):
        ori_data_info = create_data_info_after_loading()
        no_flip_transform = RandomFlip3D(flip_ratio_bev_horizontal=0.)
        always_flip_transform = RandomFlip3D(flip_ratio_bev_horizontal=1.)
        data_info = copy.deepcopy(ori_data_info)
        data_info = no_flip_transform(data_info)
        self.assertIn('pcd_horizontal_flip', data_info)
        assert_allclose(data_info['points'].tensor,
                        ori_data_info['points'].tensor)

        torch.allclose(data_info['gt_bboxes_3d'].tensor,
                       ori_data_info['gt_bboxes_3d'].tensor)
        data_info = copy.deepcopy(ori_data_info)
        data_info = always_flip_transform(data_info)
        assert_allclose(data_info['points'].tensor[:, 0],
                        ori_data_info['points'].tensor[:, 0])
        assert_allclose(data_info['points'].tensor[:, 1],
                        -ori_data_info['points'].tensor[:, 1])
        assert_allclose(data_info['points'].tensor[:, 2],
                        ori_data_info['points'].tensor[:, 2])

        assert_allclose(data_info['gt_bboxes_3d'].tensor[:, 0],
                        ori_data_info['gt_bboxes_3d'].tensor[:, 0])
        assert_allclose(data_info['gt_bboxes_3d'].tensor[:, 1],
                        -ori_data_info['gt_bboxes_3d'].tensor[:, 1])
        assert_allclose(data_info['gt_bboxes_3d'].tensor[:, 2],
                        ori_data_info['gt_bboxes_3d'].tensor[:, 2])


class TestGlobalAlignment(unittest.TestCase):

    def test_global_alignment(self):
        data_info = create_data_info_after_loading()
        global_align_transform = GlobalAlignment(rotation_axis=2)
        data_info['axis_align_matrix'] = np.array(
            [[0.945519, 0.325568, 0., -5.38439],
             [-0.325568, 0.945519, 0., -2.87178], [0., 0., 1., -0.06435],
             [0., 0., 0., 1.]],
            dtype=np.float32)
        global_align_transform(data_info)

        data_info['axis_align_matrix'] = np.array(
            [[0.945519, 0.325568, 0., -5.38439], [0, 2, 0., -2.87178],
             [0., 0., 1., -0.06435], [0., 0., 0., 1.]],
            dtype=np.float32)
        # assert the rot metric
        with self.assertRaises(AssertionError):
            global_align_transform(data_info)
