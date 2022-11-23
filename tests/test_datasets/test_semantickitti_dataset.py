# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.datasets import SemanticKITTIDataset
from mmdet3d.utils import register_all_modules


def _generate_semantickitti_dataset_config():
    data_root = './tests/data/semantickitti/'
    ann_file = 'semantickitti_infos.pkl'
    classes = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign')
    palette = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]
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
            seg_3d_dtype=np.int32),
        dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ]

    data_prefix = dict(
        pts='sequences/00/velodyne', pts_semantic_mask='sequences/00/labels')

    return (data_root, ann_file, classes, palette, data_prefix, pipeline,
            modality)


class TestSemanticKITTIDataset(unittest.TestCase):

    def test_semantickitti(self):
        data_root, ann_file, classes, palette, data_prefix, \
            pipeline, modality, = _generate_semantickitti_dataset_config()

        register_all_modules()
        np.random.seed(0)
        semantickitti_dataset = SemanticKITTIDataset(
            data_root,
            ann_file,
            metainfo=dict(classes=classes, palette=palette),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality)

        input_dict = semantickitti_dataset.prepare_data(0)

        points = input_dict['inputs']['points']
        data_sample = input_dict['data_samples']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
        self.assertEqual(points.shape[0], pts_semantic_mask.shape[0])
