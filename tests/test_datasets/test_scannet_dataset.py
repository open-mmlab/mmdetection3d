# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets import ScanNetDataset, ScanNetSegDataset
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.utils import register_all_modules


def _generate_scannet_seg_dataset_config():
    data_root = './tests/data/scannet/'
    ann_file = 'scannet_infos.pkl'
    classes = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')
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
    scene_idxs = [0]
    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            use_color=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True),
        dict(type='PointSegClassMapping'),
        dict(
            type='IndoorPatchPointSample',
            num_points=5,
            block_size=1.5,
            ignore_index=len(classes),
            use_normalized_coord=True,
            enlarge_size=0.2,
            min_unique_num=None),
        dict(type='NormalizePointsColor', color_mean=None),
        dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ]

    data_prefix = dict(
        pts='points',
        pts_instance_mask='instance_mask',
        pts_semantic_mask='semantic_mask')
    return (data_root, ann_file, classes, palette, scene_idxs, data_prefix,
            pipeline, modality)


def _generate_scannet_dataset_config():
    data_root = 'tests/data/scannet'
    ann_file = 'scannet_infos.pkl'
    classes = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')
    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_mask_3d=True,
            with_seg_3d=True),
        dict(type='GlobalAlignment', rotation_axis=2),
        dict(type='PointSegClassMapping'),
        dict(type='PointSample', num_points=5),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=1.0,
            flip_ratio_bev_vertical=1.0),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.087266, 0.087266],
            scale_ratio_range=[1.0, 1.0],
            shift_height=True),
        dict(
            type='Pack3DDetInputs',
            keys=[
                'points', 'pts_semantic_mask', 'gt_bboxes_3d', 'gt_labels_3d',
                'pts_instance_mask'
            ])
    ]
    data_prefix = dict(
        pts='points',
        pts_instance_mask='instance_mask',
        pts_semantic_mask='semantic_mask')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


class TestScanNetDataset(unittest.TestCase):

    def test_scannet(self):
        np.random.seed(0)
        data_root, ann_file, classes, data_prefix, \
            pipeline, modality, = _generate_scannet_dataset_config()
        register_all_modules()
        scannet_dataset = ScanNetDataset(
            data_root,
            ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(classes=classes),
            modality=modality)

        scannet_dataset.prepare_data(0)
        input_dict = scannet_dataset.get_data_info(0)
        scannet_dataset[0]
        # assert the the path should contains data_prefix and data_root
        self.assertIn(data_prefix['pts'],
                      input_dict['lidar_points']['lidar_path'])
        self.assertIn(data_root, input_dict['lidar_points']['lidar_path'])

        ann_info = scannet_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        except_label = np.array([
            6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0,
            0, 0, 0, 0, 0, 5, 5, 5
        ])

        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        assert_allclose(ann_info['gt_labels_3d'], except_label)
        self.assertIsInstance(ann_info['gt_bboxes_3d'], DepthInstance3DBoxes)
        assert len(ann_info['gt_bboxes_3d']) == 27
        assert torch.allclose(ann_info['gt_bboxes_3d'].tensor.sum(),
                              torch.tensor([107.7353]))

        no_class_scannet_dataset = ScanNetDataset(
            data_root, ann_file, metainfo=dict(classes=['cabinet']))

        input_dict = no_class_scannet_dataset.get_data_info(0)
        ann_info = no_class_scannet_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        self.assertIn('gt_labels_3d', ann_info)
        # assert mapping to -1 or 1
        assert (ann_info['gt_labels_3d'] <= 0).all()
        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        # all instance have been filtered by classes
        self.assertEqual(len(ann_info['gt_labels_3d']), 27)
        self.assertEqual(len(no_class_scannet_dataset.metainfo['classes']), 1)

    def test_scannet_seg(self):
        data_root, ann_file, classes, palette, scene_idxs, data_prefix, \
            pipeline, modality, = _generate_scannet_seg_dataset_config()

        register_all_modules()
        np.random.seed(0)
        scannet_seg_dataset = ScanNetSegDataset(
            data_root,
            ann_file,
            metainfo=dict(classes=classes, palette=palette),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            scene_idxs=scene_idxs)

        input_dict = scannet_seg_dataset.prepare_data(0)

        points = input_dict['inputs']['points']
        data_sample = input_dict['data_samples']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask

        expected_points = torch.tensor([[
            0.0000, 0.0000, 1.2427, 0.6118, 0.5529, 0.4471, -0.6462, -1.0046,
            0.4280
        ],
                                        [
                                            0.1553, -0.0074, 1.6077, 0.5882,
                                            0.6157, 0.5569, -0.6001, -1.0068,
                                            0.5537
                                        ],
                                        [
                                            0.1518, 0.6016, 0.6548, 0.1490,
                                            0.1059, 0.0431, -0.6012, -0.8309,
                                            0.2255
                                        ],
                                        [
                                            -0.7494, 0.1033, 0.6756, 0.5216,
                                            0.4353, 0.3333, -0.8687, -0.9748,
                                            0.2327
                                        ],
                                        [
                                            -0.6836, -0.0203, 0.5884, 0.5765,
                                            0.5020, 0.4510, -0.8491, -1.0105,
                                            0.2027
                                        ]])
        expected_pts_semantic_mask = np.array([13, 13, 12, 2, 0])

        assert torch.allclose(points, expected_points, 1e-2)
        self.assertTrue(
            (pts_semantic_mask.numpy() == expected_pts_semantic_mask).all())
