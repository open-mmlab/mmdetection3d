# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core.points import DepthPoints
from mmdet3d.datasets.pipelines import MultiScaleFlipAug3D


def test_multi_scale_flip_aug_3D():
    np.random.seed(0)
    transforms = [{
        'type': 'GlobalRotScaleTrans',
        'rot_range': [-0.1, 0.1],
        'scale_ratio_range': [0.9, 1.1],
        'translation_std': [0, 0, 0]
    }, {
        'type': 'RandomFlip3D',
        'sync_2d': False,
        'flip_ratio_bev_horizontal': 0.5
    }, {
        'type': 'PointSample',
        'num_points': 5
    }, {
        'type':
        'DefaultFormatBundle3D',
        'class_names': ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                        'dresser', 'night_stand', 'bookshelf', 'bathtub'),
        'with_label':
        False
    }, {
        'type': 'Collect3D',
        'keys': ['points']
    }]
    img_scale = (1333, 800)
    pts_scale_ratio = 1
    multi_scale_flip_aug_3D = MultiScaleFlipAug3D(transforms, img_scale,
                                                  pts_scale_ratio)
    pts_file_name = 'tests/data/sunrgbd/points/000001.bin'
    sample_idx = 4
    file_name = 'tests/data/sunrgbd/points/000001.bin'
    bbox3d_fields = []
    points = np.array([[0.20397437, 1.4267826, -1.0503972, 0.16195858],
                       [-2.2095256, 3.3159535, -0.7706928, 0.4416629],
                       [1.5090443, 3.2764456, -1.1913797, 0.02097607],
                       [-1.373904, 3.8711405, 0.8524302, 2.064786],
                       [-1.8139812, 3.538856, -1.0056694, 0.20668638]])
    points = DepthPoints(points, points_dim=4, attribute_dims=dict(height=3))
    results = dict(
        points=points,
        pts_file_name=pts_file_name,
        sample_idx=sample_idx,
        file_name=file_name,
        bbox3d_fields=bbox3d_fields)
    results = multi_scale_flip_aug_3D(results)
    expected_points = torch.tensor(
        [[-2.2418, 3.2942, -0.7707, 0.4417], [-1.4116, 3.8575, 0.8524, 2.0648],
         [-1.8484, 3.5210, -1.0057, 0.2067], [0.1900, 1.4287, -1.0504, 0.1620],
         [1.4770, 3.2910, -1.1914, 0.0210]],
        dtype=torch.float32)

    assert torch.allclose(
        results['points'][0]._data, expected_points, atol=1e-4)
