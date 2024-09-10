# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from mmdet3d.datasets import WaymoDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def _generate_waymo_dataset_config():
    data_root = 'tests/data/waymo/kitti_format'
    ann_file = 'waymo_infos_train.pkl'
    classes = ['Car', 'Pedestrian', 'Cyclist']
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

    modality = dict(use_lidar=True, use_camera=True)
    data_prefix = data_prefix = dict(
        pts='training/velodyne', CAM_FRONT='training/image_0')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


def test_getitem():
    data_root, ann_file, classes, data_prefix, \
        pipeline, modality, = _generate_waymo_dataset_config()

    waymo_dataset = WaymoDataset(
        data_root,
        ann_file,
        data_prefix=data_prefix,
        pipeline=pipeline,
        metainfo=dict(classes=classes),
        modality=modality)

    waymo_dataset.prepare_data(0)
    input_dict = waymo_dataset.get_data_info(0)
    waymo_dataset[0]
    # assert the the path should contains data_prefix and data_root
    assert data_prefix['pts'] in input_dict['lidar_points']['lidar_path']
    assert data_root in input_dict['lidar_points']['lidar_path']
    for cam_id, img_info in input_dict['images'].items():
        if 'img_path' in img_info:
            assert data_prefix['CAM_FRONT'] in img_info['img_path']
            assert data_root in img_info['img_path']

    ann_info = waymo_dataset.parse_ann_info(input_dict)

    # only one instance
    assert 'gt_labels_3d' in ann_info
    assert ann_info['gt_labels_3d'].dtype == np.int64

    assert 'gt_bboxes_3d' in ann_info
    assert isinstance(ann_info['gt_bboxes_3d'], LiDARInstance3DBoxes)
    assert torch.allclose(ann_info['gt_bboxes_3d'].tensor.sum(),
                          torch.tensor(43.3103))
    assert 'centers_2d' in ann_info
    assert ann_info['centers_2d'].dtype == np.float32
    assert 'depths' in ann_info
    assert ann_info['depths'].dtype == np.float32
