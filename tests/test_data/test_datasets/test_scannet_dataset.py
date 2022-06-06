# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.core import DepthInstance3DBoxes
from mmdet3d.datasets import ScanNetDataset


def _generate_scannet_dataset_config():
    data_root = 'tests/data/scannet'
    ann_file = 'scannet_infos.pkl'
    classes = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')
    # TODO add pipline
    from mmcv.transforms.base import BaseTransform
    from mmengine.registry import TRANSFORMS
    if 'Identity' not in TRANSFORMS:

        @TRANSFORMS.register_module()
        class Identity(BaseTransform):

            def transform(self, info):
                if 'ann_info' in info:
                    info['gt_labels_3d'] = info['ann_info']['gt_labels_3d']
                return info

    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(type='Identity'),
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

        scannet_dataset = ScanNetDataset(
            data_root,
            ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(CLASSES=classes),
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
            data_root, ann_file, metainfo=dict(CLASSES=['cabinet']))

        input_dict = no_class_scannet_dataset.get_data_info(0)
        ann_info = no_class_scannet_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        self.assertIn('gt_labels_3d', ann_info)
        # assert mapping to -1 or 1
        assert (ann_info['gt_labels_3d'] <= 0).all()
        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        # all instance have been filtered by classes
        self.assertEqual(len(ann_info['gt_labels_3d']), 27)
        self.assertEqual(len(no_class_scannet_dataset.metainfo['CLASSES']), 1)
