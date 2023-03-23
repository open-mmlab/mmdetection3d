# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets.transforms import PointSegClassMapping
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile)
from mmdet3d.structures import DepthPoints, LiDARPoints
from mmdet3d.testing import create_dummy_data_info


class TestLoadPointsFromFile(unittest.TestCase):

    def test_load_points_from_file(self):
        use_dim = 3
        backend_args = None
        load_points_transform = LoadPointsFromFile(
            coord_type='LIDAR',
            load_dim=4,
            use_dim=use_dim,
            backend_args=backend_args)
        data_info = create_dummy_data_info()
        info = load_points_transform(data_info)
        self.assertIn('points', info)
        self.assertIsInstance(info['points'], LiDARPoints)
        load_points_transform = LoadPointsFromFile(
            coord_type='DEPTH',
            load_dim=4,
            use_dim=use_dim,
            backend_args=backend_args)
        info = load_points_transform(data_info)
        self.assertIsInstance(info['points'], DepthPoints)
        self.assertEqual(info['points'].shape[-1], use_dim)
        load_points_transform = LoadPointsFromFile(
            coord_type='DEPTH',
            load_dim=4,
            use_dim=use_dim,
            shift_height=True,
            backend_args=backend_args)
        info = load_points_transform(data_info)
        # extra height dim
        self.assertEqual(info['points'].shape[-1], use_dim + 1)

        repr_str = repr(load_points_transform)
        self.assertIn('shift_height=True', repr_str)
        self.assertIn('use_color=False', repr_str)
        self.assertIn('load_dim=4', repr_str)


class TestLoadAnnotations3D(unittest.TestCase):

    def test_load_points_from_file(self):
        backend_args = None

        load_anns_transform = LoadAnnotations3D(
            with_bbox_3d=True,
            with_label_3d=True,
            with_panoptic_3d=True,
            seg_offset=2**16,
            dataset_type='semantickitti',
            seg_3d_dtype='np.uint32',
            backend_args=backend_args)
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
        self.assertIn('pts_semantic_mask', info)
        self.assertIn('pts_instance_mask', info)
        assert_allclose(
            info['pts_semantic_mask'],
            np.array([
                50, 50, 50, 70, 70, 50, 0, 50, 70, 50, 50, 70, 71, 52, 70, 50,
                50, 50, 50, 0, 50, 50, 50, 50, 50, 70, 50, 71, 50, 70, 70, 80,
                50, 70, 70, 70, 71, 70, 50, 50, 70, 50, 80, 70, 50, 70, 50, 70,
                70, 50
            ]))
        assert_allclose(
            info['pts_instance_mask'],
            np.array([
                50, 50, 50, 70, 70, 50, 0, 50, 70, 50, 50, 70, 71, 52, 70, 50,
                50, 50, 50, 0, 50, 50, 50, 50, 50, 70, 50, 71, 50, 70, 70, 80,
                50, 70, 70, 70, 71, 70, 50, 50, 70, 50, 80, 70, 50, 70, 50, 70,
                70, 50
            ]))
        repr_str = repr(load_anns_transform)
        self.assertIn('with_bbox_3d=True', repr_str)
        self.assertIn('with_label_3d=True', repr_str)
        self.assertIn('with_bbox_depth=False', repr_str)
        self.assertIn('with_panoptic_3d=True', repr_str)


class TestPointSegClassMapping(unittest.TestCase):

    def test_point_seg_class_mapping(self):
        results = dict()
        results['pts_semantic_mask'] = np.array([1, 2, 3, 4, 5])
        results['seg_label_mapping'] = np.array([3, 0, 1, 2, 3, 3])
        point_seg_mapping_transform = PointSegClassMapping()
        results = point_seg_mapping_transform(results)
        assert_allclose(results['pts_semantic_mask'], np.array([0, 1, 2, 3,
                                                                3]))
