# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets import S3DISDataset, S3DISSegDataset
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.utils import register_all_modules


def _generate_s3dis_seg_dataset_config():
    data_root = './tests/data/s3dis/'
    ann_file = 's3dis_infos.pkl'
    classes = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
    palette = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
               [255, 0, 255], [100, 100, 255], [200, 200, 100],
               [170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100],
               [200, 200, 200], [50, 50, 50]]
    scene_idxs = [0 for _ in range(20)]
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
            block_size=1.0,
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


def _generate_s3dis_dataset_config():
    data_root = 'tests/data/s3dis'
    ann_file = 's3dis_infos.pkl'
    classes = ('table', 'chair', 'sofa', 'bookcase', 'board')
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
            with_bbox_3d=True,
            with_label_3d=True,
            with_mask_3d=True,
            with_seg_3d=True),
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
            scale_ratio_range=[1.0, 1.0]),
        dict(type='NormalizePointsColor', color_mean=None),
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


class TestS3DISDataset(unittest.TestCase):

    def test_s3dis(self):
        np.random.seed(0)
        data_root, ann_file, classes, data_prefix, \
            pipeline, modality = _generate_s3dis_dataset_config()
        register_all_modules()
        s3dis_dataset = S3DISDataset(
            data_root,
            ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(classes=classes),
            modality=modality)

        s3dis_dataset.prepare_data(0)
        input_dict = s3dis_dataset.get_data_info(0)
        s3dis_dataset[0]
        # assert the path should contains data_prefix and data_root
        self.assertIn(data_prefix['pts'],
                      input_dict['lidar_points']['lidar_path'])
        self.assertIn(data_root, input_dict['lidar_points']['lidar_path'])

        ann_info = s3dis_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        except_label = np.array([1, 1, 3, 1, 2, 0, 0, 0, 3])

        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        assert_allclose(ann_info['gt_labels_3d'], except_label)
        self.assertIsInstance(ann_info['gt_bboxes_3d'], DepthInstance3DBoxes)
        assert len(ann_info['gt_bboxes_3d']) == 9
        assert torch.allclose(ann_info['gt_bboxes_3d'].tensor.sum(),
                              torch.tensor([63.0455]))

        no_class_s3dis_dataset = S3DISDataset(
            data_root, ann_file, metainfo=dict(classes=['table']))

        input_dict = no_class_s3dis_dataset.get_data_info(0)
        ann_info = no_class_s3dis_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        self.assertIn('gt_labels_3d', ann_info)
        # assert mapping to -1 or 1
        assert (ann_info['gt_labels_3d'] <= 0).all()
        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        # all instance have been filtered by classes
        self.assertEqual(len(ann_info['gt_labels_3d']), 9)
        self.assertEqual(len(no_class_s3dis_dataset.metainfo['classes']), 1)

    def test_s3dis_seg(self):
        data_root, ann_file, classes, palette, scene_idxs, data_prefix, \
            pipeline, modality, = _generate_s3dis_seg_dataset_config()

        register_all_modules()
        np.random.seed(0)

        s3dis_seg_dataset = S3DISSegDataset(
            data_root,
            ann_file,
            metainfo=dict(classes=classes, palette=palette),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            scene_idxs=scene_idxs)

        input_dict = s3dis_seg_dataset.prepare_data(0)

        points = input_dict['inputs']['points']
        data_sample = input_dict['data_samples']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask

        expected_points = torch.tensor([[
            0.0000, 0.0000, 3.1720, 0.4706, 0.4431, 0.3725, 0.4624, 0.7502,
            0.9543
        ],
                                        [
                                            0.2880, -0.5900, 0.0650, 0.3451,
                                            0.3373, 0.3490, 0.5119, 0.5518,
                                            0.0196
                                        ],
                                        [
                                            0.1570, 0.6000, 3.1700, 0.4941,
                                            0.4667, 0.3569, 0.4893, 0.9519,
                                            0.9537
                                        ],
                                        [
                                            -0.1320, 0.3950, 0.2720, 0.3216,
                                            0.2863, 0.2275, 0.4397, 0.8830,
                                            0.0818
                                        ],
                                        [
                                            -0.4860, -0.0640, 3.1710, 0.3843,
                                            0.3725, 0.3059, 0.3789, 0.7286,
                                            0.9540
                                        ]])

        expected_pts_semantic_mask = np.array([0, 1, 0, 8, 0])

        assert torch.allclose(points, expected_points, 1e-2)
        self.assertTrue(
            (pts_semantic_mask.numpy() == expected_pts_semantic_mask).all())
