# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets import (GlobalAlignment, RandomFlip3D,
                              SemanticKittiDataset)
from mmdet3d.datasets.transforms import GlobalRotScaleTrans, LaserMix, PolarMix
from mmdet3d.structures import LiDARPoints
from mmdet3d.testing import create_data_info_after_loading
from mmdet3d.utils import register_all_modules

register_all_modules()


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


class TestPolarMix(unittest.TestCase):

    def setUp(self):
        self.pre_transform = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True,
                seg_3d_dtype='np.int32'),
            dict(type='PointSegClassMapping'),
        ]
        classes = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                   'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                   'other-ground', 'building', 'fence', 'vegetation', 'trunck',
                   'terrian', 'pole', 'traffic-sign')
        seg_label_mapping = {
            0: 0,  # "unlabeled"
            1: 0,  # "outlier" mapped to "unlabeled" --------------mapped
            10: 1,  # "car"
            11: 2,  # "bicycle"
            13: 5,  # "bus" mapped to "other-vehicle" --------------mapped
            15: 3,  # "motorcycle"
            16: 5,  # "on-rails" mapped to "other-vehicle" ---------mapped
            18: 4,  # "truck"
            20: 5,  # "other-vehicle"
            30: 6,  # "person"
            31: 7,  # "bicyclist"
            32: 8,  # "motorcyclist"
            40: 9,  # "road"
            44: 10,  # "parking"
            48: 11,  # "sidewalk"
            49: 12,  # "other-ground"
            50: 13,  # "building"
            51: 14,  # "fence"
            52: 0,  # "other-structure" mapped to "unlabeled" ------mapped
            60: 9,  # "lane-marking" to "road" ---------------------mapped
            70: 15,  # "vegetation"
            71: 16,  # "trunk"
            72: 17,  # "terrain"
            80: 18,  # "pole"
            81: 19,  # "traffic-sign"
            99: 0,  # "other-object" to "unlabeled" ----------------mapped
            252: 1,  # "moving-car" to "car" ------------------------mapped
            253: 7,  # "moving-bicyclist" to "bicyclist" ------------mapped
            254: 6,  # "moving-person" to "person" ------------------mapped
            255: 8,  # "moving-motorcyclist" to "motorcyclist" ------mapped
            256: 5,  # "moving-on-rails" mapped to "other-vehic------mapped
            257: 5,  # "moving-bus" mapped to "other-vehicle" -------mapped
            258: 4,  # "moving-truck" to "truck" --------------------mapped
            259: 5  # "moving-other"-vehicle to "other-vehicle"-----mapped
        }
        max_label = 259
        self.dataset = SemanticKittiDataset(
            './tests/data/semantickitti/',
            'semantickitti_infos.pkl',
            metainfo=dict(
                classes=classes,
                seg_label_mapping=seg_label_mapping,
                max_label=max_label),
            data_prefix=dict(
                pts='sequences/00/velodyne',
                pts_semantic_mask='sequences/00/labels'),
            pipeline=[],
            modality=dict(use_lidar=True, use_camera=False))
        points = np.random.random((100, 4))
        self.results = {
            'points': LiDARPoints(points, points_dim=4),
            'pts_semantic_mask': np.random.randint(0, 20, (100, )),
            'dataset': self.dataset
        }

    def test_transform(self):
        # test assertion for invalid instance_classes
        with self.assertRaises(AssertionError):
            transform = PolarMix(instance_classes=1)

        with self.assertRaises(AssertionError):
            transform = PolarMix(instance_classes=[1.0, 2.0])

        transform = PolarMix(
            instance_classes=[15, 16, 17],
            swap_ratio=1.0,
            pre_transform=self.pre_transform)
        results = transform.transform(copy.deepcopy(self.results))
        self.assertTrue(results['points'].shape[0] ==
                        results['pts_semantic_mask'].shape[0])


class TestLaserMix(unittest.TestCase):

    def setUp(self):
        self.pre_transform = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True,
                seg_3d_dtype='np.int32'),
            dict(type='PointSegClassMapping'),
        ]
        classes = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                   'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                   'other-ground', 'building', 'fence', 'vegetation', 'trunck',
                   'terrian', 'pole', 'traffic-sign')
        seg_label_mapping = {
            0: 0,  # "unlabeled"
            1: 0,  # "outlier" mapped to "unlabeled" --------------mapped
            10: 1,  # "car"
            11: 2,  # "bicycle"
            13: 5,  # "bus" mapped to "other-vehicle" --------------mapped
            15: 3,  # "motorcycle"
            16: 5,  # "on-rails" mapped to "other-vehicle" ---------mapped
            18: 4,  # "truck"
            20: 5,  # "other-vehicle"
            30: 6,  # "person"
            31: 7,  # "bicyclist"
            32: 8,  # "motorcyclist"
            40: 9,  # "road"
            44: 10,  # "parking"
            48: 11,  # "sidewalk"
            49: 12,  # "other-ground"
            50: 13,  # "building"
            51: 14,  # "fence"
            52: 0,  # "other-structure" mapped to "unlabeled" ------mapped
            60: 9,  # "lane-marking" to "road" ---------------------mapped
            70: 15,  # "vegetation"
            71: 16,  # "trunk"
            72: 17,  # "terrain"
            80: 18,  # "pole"
            81: 19,  # "traffic-sign"
            99: 0,  # "other-object" to "unlabeled" ----------------mapped
            252: 1,  # "moving-car" to "car" ------------------------mapped
            253: 7,  # "moving-bicyclist" to "bicyclist" ------------mapped
            254: 6,  # "moving-person" to "person" ------------------mapped
            255: 8,  # "moving-motorcyclist" to "motorcyclist" ------mapped
            256: 5,  # "moving-on-rails" mapped to "other-vehic------mapped
            257: 5,  # "moving-bus" mapped to "other-vehicle" -------mapped
            258: 4,  # "moving-truck" to "truck" --------------------mapped
            259: 5  # "moving-other"-vehicle to "other-vehicle"-----mapped
        }
        max_label = 259
        self.dataset = SemanticKittiDataset(
            './tests/data/semantickitti/',
            'semantickitti_infos.pkl',
            metainfo=dict(
                classes=classes,
                seg_label_mapping=seg_label_mapping,
                max_label=max_label),
            data_prefix=dict(
                pts='sequences/00/velodyne',
                pts_semantic_mask='sequences/00/labels'),
            pipeline=[],
            modality=dict(use_lidar=True, use_camera=False))
        points = np.random.random((100, 4))
        self.results = {
            'points': LiDARPoints(points, points_dim=4),
            'pts_semantic_mask': np.random.randint(0, 20, (100, )),
            'dataset': self.dataset
        }

    def test_transform(self):
        # test assertion for invalid num_areas
        with self.assertRaises(AssertionError):
            transform = LaserMix(num_areas=3, pitch_angles=[-20, 0])

        with self.assertRaises(AssertionError):
            transform = LaserMix(num_areas=[3.0, 4.0], pitch_angles=[-20, 0])

        # test assertion for invalid pitch_angles
        with self.assertRaises(AssertionError):
            transform = LaserMix(num_areas=[3, 4], pitch_angles=[-20])

        with self.assertRaises(AssertionError):
            transform = LaserMix(num_areas=[3, 4], pitch_angles=[0, -20])

        transform = LaserMix(
            num_areas=[3, 4, 5, 6],
            pitch_angles=[-20, 0],
            pre_transform=self.pre_transform)
        results = transform.transform(copy.deepcopy(self.results))
        self.assertTrue(results['points'].shape[0] ==
                        results['pts_semantic_mask'].shape[0])
