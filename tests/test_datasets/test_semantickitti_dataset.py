# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.datasets import SemanticKittiDataset
from mmdet3d.utils import register_all_modules


def _generate_semantickitti_dataset_config():
    data_root = './tests/data/semantickitti/'
    ann_file = 'semantickitti_infos.pkl'
    classes = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
               'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
               'other-ground', 'building', 'fence', 'vegetation', 'trunck',
               'terrian', 'pole', 'traffic-sign')

    seg_label_mapping = {
        0: 19,  # "unlabeled"
        1: 19,  # "outlier" mapped to "unlabeled" --------------mapped
        10: 0,  # "car"
        11: 1,  # "bicycle"
        13: 4,  # "bus" mapped to "other-vehicle" --------------mapped
        15: 2,  # "motorcycle"
        16: 4,  # "on-rails" mapped to "other-vehicle" ---------mapped
        18: 3,  # "truck"
        20: 4,  # "other-vehicle"
        30: 5,  # "person"
        31: 6,  # "bicyclist"
        32: 7,  # "motorcyclist"
        40: 8,  # "road"
        44: 9,  # "parking"
        48: 10,  # "sidewalk"
        49: 11,  # "other-ground"
        50: 12,  # "building"
        51: 13,  # "fence"
        52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
        60: 8,  # "lane-marking" to "road" ---------------------mapped
        70: 14,  # "vegetation"
        71: 15,  # "trunk"
        72: 16,  # "terrain"
        80: 17,  # "pole"
        81: 18,  # "traffic-sign"
        99: 19,  # "other-object" to "unlabeled" ----------------mapped
        252: 0,  # "moving-car" to "car" ------------------------mapped
        253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
        254: 5,  # "moving-person" to "person" ------------------mapped
        255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
        256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
        257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
        258: 3,  # "moving-truck" to "truck" --------------------mapped
        259: 4  # "moving-other"-vehicle to "other-vehicle"-----mapped
    }
    max_label = 259
    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=4,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            seg_3d_dtype='np.int32'),
        dict(type='PointSegClassMapping'),
        dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ]

    data_prefix = dict(
        pts='sequences/00/velodyne', pts_semantic_mask='sequences/00/labels')

    return (data_root, ann_file, classes, data_prefix, pipeline, modality,
            seg_label_mapping, max_label)


class TestSemanticKittiDataset(unittest.TestCase):

    def test_semantickitti(self):
        (data_root, ann_file, classes, data_prefix, pipeline, modality,
         seg_label_mapping,
         max_label) = _generate_semantickitti_dataset_config()

        register_all_modules()
        np.random.seed(0)
        semantickitti_dataset = SemanticKittiDataset(
            data_root,
            ann_file,
            metainfo=dict(
                classes=classes,
                seg_label_mapping=seg_label_mapping,
                max_label=max_label),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality)

        input_dict = semantickitti_dataset.prepare_data(0)

        points = input_dict['inputs']['points']
        data_sample = input_dict['data_samples']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
        self.assertEqual(points.shape[0], pts_semantic_mask.shape[0])

        expected_pts_semantic_mask = np.array([
            12, 12, 12, 14, 14, 12, 19, 12, 14, 12, 12, 14, 15, 19, 14, 12, 12,
            12, 12, 19, 12, 12, 12, 12, 12, 14, 12, 15, 12, 14, 14, 17, 12, 14,
            14, 14, 15, 14, 12, 12, 14, 12, 17, 14, 12, 14, 12, 14, 14, 12
        ])

        self.assertTrue(
            (pts_semantic_mask.numpy() == expected_pts_semantic_mask).all())
