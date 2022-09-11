# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.datasets import SensatUrbanDataset
from mmdet3d.utils import register_all_modules


def _generate_sensaturban_dataset_config():
    data_root = '../data/sensaturban/'
    ann_file = 'sensaturban_infos.pkl'
    classes = ('Ground', 'Vegetation', 'Building', 'Wall', 'Bridge', 'Parking',
               'Rail', 'Traffic', 'Street', 'Car', 'Footpath', 'Bike', 'Water')
    palette = [[152, 223, 138], [31, 119, 180], [255, 187,
                                                 120], [188, 189, 34],
               [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213],
               [148, 103, 189], [196, 156, 148], [23, 190, 207],
               [247, 182, 210], [219, 219, 141]]
    modality = dict(use_lidar=True, use_camera=True)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadImageFromFile',
            color_type='color',
            imdecode_backend='cv2'),
        dict(
            type='LoadDepthFromFile',
            with_transform=False,
            imdecode_backend='cv2'),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            with_seg=True,
            seg_3d_dtype=np.int8),
        dict(
            type='Pack3DDetInputs',
            keys=[
                'points', 'img', 'depth_img', 'pts_semantic_mask', 'gt_seg_map'
            ])
    ]

    data_prefix = dict(
        pts_prefix='train/points',
        pts_semantic_mask_prefix='train/labels',
        bev_prefix='train/bevs',
        alt_prefix='train/altitude',
        bev_semantic_mask_prefix='train/masks')

    return (data_root, ann_file, classes, palette, data_prefix, pipeline,
            modality)


class TestSensatUrbanDataset(unittest.TestCase):

    def test_sensaturban(self):
        data_root, ann_file, classes, palette, data_prefix, \
         pipeline, modality, = _generate_sensaturban_dataset_config()

        register_all_modules()
        np.random.seed(0)
        sensaturban_dataset = SensatUrbanDataset(
            data_root,
            ann_file,
            metainfo=dict(CLASSES=classes, PALETTE=palette),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality)

        input_dict = sensaturban_dataset.prepare_data(0)

        points = input_dict['inputs']['points']
        bev = input_dict['inputs']['img']
        altitude = input_dict['inputs']['depth_img']
        data_sample = input_dict['data_sample']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
        seg_map = data_sample.gt_pts_seg.seg_map

        self.assertEqual(bev.shape[1], altitude.shape[1])
        self.assertEqual(bev.shape[1], seg_map.shape[1])
        self.assertEqual(bev.shape[2], altitude.shape[2])
        self.assertEqual(bev.shape[2], seg_map.shape[2])
        self.assertEqual(points.shape[0], pts_semantic_mask.shape[0])
