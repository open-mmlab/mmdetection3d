# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.testing import assert_allclose
from utils import create_dummy_data_info

from mmdet3d.core import DepthPoints, LiDARPoints
from mmdet3d.datasets.pipelines.loading import (LoadAnnotations3D,
                                                LoadPointsFromFile)


class TestLoadPointsFromFile(unittest.TestCase):

    def test_load_points_from_file(self):
        use_dim = 3
        file_client_args = dict(backend='disk')
        load_points_transform = LoadPointsFromFile(
            coord_type='LIDAR',
            load_dim=4,
            use_dim=use_dim,
            file_client_args=file_client_args)
        data_info = create_dummy_data_info()
        info = load_points_transform(data_info)
        self.assertIn('points', info)
        self.assertIsInstance(info['points'], LiDARPoints)
        load_points_transform = LoadPointsFromFile(
            coord_type='DEPTH',
            load_dim=4,
            use_dim=use_dim,
            file_client_args=file_client_args)
        info = load_points_transform(data_info)
        self.assertIsInstance(info['points'], DepthPoints)
        self.assertEqual(info['points'].shape[-1], use_dim)
        load_points_transform = LoadPointsFromFile(
            coord_type='DEPTH',
            load_dim=4,
            use_dim=use_dim,
            shift_height=True,
            file_client_args=file_client_args)
        info = load_points_transform(data_info)
        # extra height dim
        self.assertEqual(info['points'].shape[-1], use_dim + 1)

        repr_str = repr(load_points_transform)
        self.assertIn('shift_height=True', repr_str)
        self.assertIn('use_color=False', repr_str)
        self.assertIn('load_dim=4', repr_str)


class TestLoadAnnotations3D(unittest.TestCase):

    def test_load_points_from_file(self):
        file_client_args = dict(backend='disk')

        load_anns_transform = LoadAnnotations3D(
            with_bbox_3d=True,
            with_label_3d=True,
            file_client_args=file_client_args)
        self.assertIs(load_anns_transform.with_seg, False)
        self.assertIs(load_anns_transform.with_bbox_3d, True)
        self.assertIs(load_anns_transform.with_label_3d, True)
        data_info = create_dummy_data_info()
        info = load_anns_transform(data_info)
        self.assertIn('gt_bboxes_3d', info)
        assert_allclose(info['gt_bboxes_3d'].tensor.sum(),
                        torch.tensor(7.2650))
        self.assertIn('gt_labels_3d', info)
        assert_allclose(info['gt_labels_3d'], torch.tensor([1]))
        repr_str = repr(load_anns_transform)
        self.assertIn('with_bbox_3d=True', repr_str)
        self.assertIn('with_label_3d=True', repr_str)
        self.assertIn('with_bbox_depth=False', repr_str)
