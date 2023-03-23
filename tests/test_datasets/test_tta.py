# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest

from mmdet3d.datasets.transforms import *  # noqa
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import LiDARPoints


class TestMuitiScaleFlipAug3D(TestCase):

    def test_exception(self):
        with pytest.raises(TypeError):
            tta_transform = dict(
                type='TestTimeAug',
                transforms=[
                    dict(
                        type='mmdet3d.RandomFlip3D',
                        flip_ratio_bev_horizontal=0.0,
                        flip_ratio_bev_vertical=0.0)
                ])
            TRANSFORMS.build(tta_transform)

    def test_multi_scale_flip_aug(self):
        tta_transform = dict(
            type='TestTimeAug',
            transforms=[[
                dict(
                    type='mmdet3d.RandomFlip3D',
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='mmdet3d.RandomFlip3D',
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=1.0),
                dict(
                    type='mmdet3d.RandomFlip3D',
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='mmdet3d.RandomFlip3D',
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=1.0)
            ], [dict(type='mmdet3d.Pack3DDetInputs', keys=['points'])]])
        tta_module = TRANSFORMS.build(tta_transform)

        results = dict()
        points = LiDARPoints(np.random.random((100, 4)), 4)
        results['points'] = points

        tta_results = tta_module(results.copy())
        assert [
            data_sample.metainfo['pcd_horizontal_flip']
            for data_sample in tta_results['data_samples']
        ] == [False, False, True, True]
        assert [
            data_sample.metainfo['pcd_vertical_flip']
            for data_sample in tta_results['data_samples']
        ] == [False, True, False, True]

        # TODO: support rotate and scale TTA module
