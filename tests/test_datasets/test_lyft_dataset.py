# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from mmdet3d.datasets import LyftDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def _generate_nus_dataset_config():
    data_root = 'tests/data/lyft'
    ann_file = 'lyft_infos.pkl'
    classes = [
        'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
        'motorcycle', 'bicycle', 'pedestrian', 'animal'
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
    modality = dict(use_lidar=True, use_camera=False)
    data_prefix = dict(pts='lidar', img='', sweeps='sweeps/LIDAR_TOP')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


def test_getitem():
    np.random.seed(0)
    data_root, ann_file, classes, data_prefix, pipeline, modality = \
        _generate_nus_dataset_config()

    lyft_dataset = LyftDataset(
        data_root,
        ann_file,
        data_prefix=data_prefix,
        pipeline=pipeline,
        metainfo=dict(classes=classes),
        modality=modality)

    lyft_dataset.prepare_data(0)
    input_dict = lyft_dataset.get_data_info(0)
    # assert the the path should contains data_prefix and data_root
    assert data_prefix['pts'] in input_dict['lidar_points']['lidar_path']
    assert data_root in input_dict['lidar_points']['lidar_path']

    ann_info = lyft_dataset.parse_ann_info(input_dict)

    # assert the keys in ann_info and the type
    assert 'gt_labels_3d' in ann_info
    assert ann_info['gt_labels_3d'].dtype == np.int64
    assert len(ann_info['gt_labels_3d']) == 3

    assert 'gt_bboxes_3d' in ann_info
    assert isinstance(ann_info['gt_bboxes_3d'], LiDARInstance3DBoxes)

    assert len(lyft_dataset.metainfo['classes']) == 9
