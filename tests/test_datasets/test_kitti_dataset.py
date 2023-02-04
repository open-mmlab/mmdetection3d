# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from mmdet3d.datasets import KittiDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def _generate_kitti_dataset_config():
    data_root = 'tests/data/kitti'
    ann_file = 'kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    # wait for pipline refactor

    if 'Identity' not in TRANSFORMS:

        @TRANSFORMS.register_module()
        class Identity(BaseTransform):

            def transform(self, info):
                if 'ann_info' in info:
                    info['gt_labels_3d'] = info['ann_info']['gt_labels_3d']
                data_sample = Det3DDataSample()
                gt_instances_3d = InstanceData()
                gt_instances_3d.labels_3d = info['gt_labels_3d']
                data_sample.gt_instances_3d = gt_instances_3d
                info['data_samples'] = data_sample
                return info

    pipeline = [
        dict(type='Identity'),
    ]

    modality = dict(use_lidar=True, use_camera=False)
    data_prefix = dict(pts='training/velodyne_reduced', img='training/image_2')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


def test_getitem():
    np.random.seed(0)
    data_root, ann_file, classes, data_prefix, \
        pipeline, modality, = _generate_kitti_dataset_config()
    modality['use_camera'] = True

    kitti_dataset = KittiDataset(
        data_root,
        ann_file,
        data_prefix=dict(
            pts='training/velodyne_reduced',
            img='training/image_2',
        ),
        pipeline=pipeline,
        metainfo=dict(classes=classes),
        modality=modality)

    kitti_dataset.prepare_data(0)
    input_dict = kitti_dataset.get_data_info(0)
    kitti_dataset[0]
    # assert the the path should contains data_prefix and data_root
    assert data_prefix['pts'] in input_dict['lidar_points']['lidar_path']
    assert data_root in input_dict['lidar_points']['lidar_path']
    for cam_id, img_info in input_dict['images'].items():
        if 'img_path' in img_info:
            assert data_prefix['img'] in img_info['img_path']
            assert data_root in img_info['img_path']

    ann_info = kitti_dataset.parse_ann_info(input_dict)

    # assert the keys in ann_info and the type
    assert 'instances' in ann_info

    # only one instance
    assert 'gt_labels_3d' in ann_info
    assert ann_info['gt_labels_3d'].dtype == np.int64

    assert 'gt_bboxes_3d' in ann_info
    assert isinstance(ann_info['gt_bboxes_3d'], LiDARInstance3DBoxes)
    assert torch.allclose(ann_info['gt_bboxes_3d'].tensor.sum(),
                          torch.tensor(7.2650))
    assert 'centers_2d' in ann_info
    assert ann_info['centers_2d'].dtype == np.float32
    assert 'depths' in ann_info
    assert ann_info['depths'].dtype == np.float32

    car_kitti_dataset = KittiDataset(
        data_root,
        ann_file,
        data_prefix=dict(
            pts='training/velodyne_reduced',
            img='training/image_2',
        ),
        pipeline=pipeline,
        metainfo=dict(classes=['Car']),
        modality=modality)

    input_dict = car_kitti_dataset.get_data_info(0)
    ann_info = car_kitti_dataset.parse_ann_info(input_dict)

    # assert the keys in ann_info and the type
    assert 'instances' in ann_info
    assert ann_info['gt_labels_3d'].dtype == np.int64
    # all instance have been filtered by classes
    assert len(ann_info['gt_labels_3d']) == 0
    assert len(car_kitti_dataset.metainfo['classes']) == 1
