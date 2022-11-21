# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets import SUNRGBDDataset
from mmdet3d.structures import DepthInstance3DBoxes


def _generate_scannet_dataset_config():
    data_root = 'tests/data/sunrgbd'
    ann_file = 'sunrgbd_infos.pkl'

    classes = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

    from mmcv.transforms.base import BaseTransform
    from mmengine.registry import TRANSFORMS

    if 'Identity' not in TRANSFORMS:

        @TRANSFORMS.register_module()
        class Identity(BaseTransform):

            def transform(self, info):
                if 'ann_info' in info:
                    info['gt_labels_3d'] = info['ann_info']['gt_labels_3d']
                return info

    modality = dict(use_camera=True, use_lidar=True)
    pipeline = [
        dict(type='Identity'),
    ]
    data_prefix = dict(pts='points', img='sunrgbd_trainval')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


class TestScanNetDataset(unittest.TestCase):

    def test_sunrgbd_ataset(self):
        np.random.seed(0)
        data_root, ann_file, classes, data_prefix, \
            pipeline, modality, = _generate_scannet_dataset_config()
        scannet_dataset = SUNRGBDDataset(
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
        assert data_prefix['pts'] in input_dict['lidar_points']['lidar_path']
        assert data_root in input_dict['lidar_points']['lidar_path']
        for cam_id, img_info in input_dict['images'].items():
            if 'img_path' in img_info:
                assert data_prefix['img'] in img_info['img_path']
                assert data_root in img_info['img_path']

        ann_info = scannet_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        except_label = np.array([0, 7, 6])

        self.assertEqual(ann_info['gt_labels_3d'].dtype, np.int64)
        assert_allclose(ann_info['gt_labels_3d'], except_label)
        self.assertIsInstance(ann_info['gt_bboxes_3d'], DepthInstance3DBoxes)

        self.assertEqual(len(ann_info['gt_bboxes_3d']), 3)
        assert_allclose(ann_info['gt_bboxes_3d'].tensor.sum(),
                        torch.tensor(19.2575))

        classes = ['bed']
        bed_scannet_dataset = SUNRGBDDataset(
            data_root,
            ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(classes=classes),
            modality=modality)

        input_dict = bed_scannet_dataset.get_data_info(0)
        ann_info = bed_scannet_dataset.parse_ann_info(input_dict)

        # assert the keys in ann_info and the type
        self.assertIn('gt_labels_3d', ann_info)
        # assert mapping to -1 or 1
        assert (ann_info['gt_labels_3d'] <= 0).all()
        assert ann_info['gt_labels_3d'].dtype == np.int64
        # all instance have been filtered by classes
        self.assertEqual(len(ann_info['gt_labels_3d']), 3)
        self.assertEqual(len(bed_scannet_dataset.metainfo['classes']), 1)
