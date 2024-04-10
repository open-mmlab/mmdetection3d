# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from mmdet3d.datasets import NuScenesDataset, NuScenesSegDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules


def _generate_nus_dataset_config():
    data_root = 'tests/data/nuscenes'
    ann_file = 'nus_info.pkl'
    classes = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    if 'Identity' not in TRANSFORMS:

        @TRANSFORMS.register_module()
        class Identity(BaseTransform):

            def transform(self, info):
                packed_input = dict(data_samples=Det3DDataSample())
                if 'ann_info' in info:
                    packed_input[
                        'data_samples'].gt_instances_3d = InstanceData()
                    packed_input[
                        'data_samples'].gt_instances_3d.labels_3d = info[
                            'ann_info']['gt_labels_3d']
                return packed_input

    pipeline = [
        dict(type='Identity'),
    ]
    modality = dict(use_lidar=True, use_camera=True)
    data_prefix = dict(
        pts='samples/LIDAR_TOP',
        img='samples/CAM_BACK_LEFT',
        sweeps='sweeps/LIDAR_TOP')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


def _generate_nus_seg_dataset_config():
    data_root = './tests/data/nuscenes'
    ann_file = 'nus_info.pkl'
    classes = ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation')
    seg_label_mapping = {
        0: 16,
        1: 16,
        2: 6,
        3: 6,
        4: 6,
        5: 16,
        6: 6,
        7: 16,
        8: 16,
        9: 0,
        10: 16,
        11: 16,
        12: 7,
        13: 16,
        14: 1,
        15: 2,
        16: 2,
        17: 3,
        18: 4,
        19: 16,
        20: 16,
        21: 5,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        26: 12,
        27: 13,
        28: 14,
        29: 16,
        30: 15,
        31: 16
    }
    max_label = 31
    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=5,
            use_dim=4),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            seg_3d_dtype='np.uint8'),
        dict(type='PointSegClassMapping'),
        dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ]
    data_prefix = dict(
        pts='samples/LIDAR_TOP', pts_semantic_mask='lidarseg/v1.0-trainval')

    return (data_root, ann_file, classes, data_prefix, pipeline, modality,
            seg_label_mapping, max_label)


class TestNuScenesDataset(unittest.TestCase):

    def test_nuscenes(self):
        np.random.seed(0)
        data_root, ann_file, classes, data_prefix, pipeline, modality = \
            _generate_nus_dataset_config()

        nus_dataset = NuScenesDataset(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(classes=classes),
            modality=modality)

        nus_dataset.prepare_data(0)
        input_dict = nus_dataset.get_data_info(0)
        # assert the path should contains data_prefix and data_root
        self.assertIn(data_prefix['pts'],
                      input_dict['lidar_points']['lidar_path'])
        self.assertIn(data_root, input_dict['lidar_points']['lidar_path'])

        for cam_id, img_info in input_dict['images'].items():
            if 'img_path' in img_info:
                self.assertIn(data_prefix['img'], img_info['img_path'])
                self.assertIn(data_root, img_info['img_path'])

        ann_info = nus_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        self.assertIn('gt_labels_3d', ann_info)
        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        assert len(ann_info['gt_labels_3d']) == 37

        self.assertIn('gt_bboxes_3d', ann_info)
        self.assertIsInstance(ann_info['gt_bboxes_3d'], LiDARInstance3DBoxes)

        assert len(nus_dataset.metainfo['classes']) == 10

        self.assertEqual(input_dict['token'],
                         'fd8420396768425eabec9bdddf7e64b6')
        self.assertEqual(input_dict['timestamp'], 1533201470.448696)

    def test_nuscenes_seg(self):
        data_root, ann_file, classes, data_prefix, pipeline, modality, \
            seg_label_mapping, max_label = _generate_nus_seg_dataset_config()

        register_all_modules()
        np.random.seed(0)

        nus_seg_dataset = NuScenesSegDataset(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(
                classes=classes,
                seg_label_mapping=seg_label_mapping,
                max_label=max_label),
            modality=modality)

        expected_pts_semantic_mask = np.array([
            10, 10, 14, 14, 10, 16, 14, 10, 16, 14, 10, 10, 10, 10, 13, 10, 14,
            14, 10, 16, 14, 3, 16, 14, 16, 10, 10, 16, 16, 10, 10, 14, 16, 10,
            15, 14, 14, 14, 16, 3
        ])

        input_dict = nus_seg_dataset.prepare_data(0)
        points = input_dict['inputs']['points']
        data_sample = input_dict['data_samples']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
        self.assertEqual(points.shape[0], pts_semantic_mask.shape[0])
        self.assertTrue(
            (pts_semantic_mask.numpy() == expected_pts_semantic_mask).all())
