# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
from mmengine import DefaultScope

from mmdet3d.datasets.transforms import *  # noqa
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import LiDARPoints

DefaultScope.get_instance('test_multi_scale_flip_aug_3d', scope_name='mmdet3d')


class TestMuitiScaleFlipAug3D(TestCase):

    def test_exception(self):
        with pytest.raises(TypeError):
            tta_transform = dict(
                type='TestTimeAug',
                transforms=[
                    dict(
                        type='RandomFlip3D',
                        flip_ratio_bev_horizontal=0.0,
                        flip_ratio_bev_vertical=0.0)
                ])
            TRANSFORMS.build(tta_transform)

    def test_multi_scale_flip_aug(self):
        tta_transform = dict(
            type='TestTimeAug',
            transforms=[[
                dict(
                    type='RandomFlip3D',
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='RandomFlip3D',
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=1.0),
                dict(
                    type='RandomFlip3D',
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='RandomFlip3D',
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=1.0)
            ], [dict(type='Pack3DDetInputs', keys=['points'])]])
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

        tta_transform = dict(
            type='TestTimeAug',
            transforms=[[
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, -0.78539816],
                    scale_ratio_range=[1.0, 1.0],
                    translation_std=[0, 0, 0]),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1.0, 1.0],
                    translation_std=[0, 0, 0]),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0.78539816, 0.78539816],
                    scale_ratio_range=[1.0, 1.0],
                    translation_std=[0, 0, 0])
            ], [dict(type='Pack3DDetInputs', keys=['points'])]])
        tta_module = TRANSFORMS.build(tta_transform)

        results = dict()
        points = LiDARPoints(np.random.random((100, 4)), 4)
        results['points'] = points

        tta_results = tta_module(results.copy())
        assert [
            data_sample.metainfo['pcd_rotation_angle']
            for data_sample in tta_results['data_samples']
        ] == [-0.78539816, 0, 0.78539816]
        assert [
            data_sample.metainfo['pcd_scale_factor']
            for data_sample in tta_results['data_samples']
        ] == [1.0, 1.0, 1.0]

        tta_transform = dict(
            type='TestTimeAug',
            transforms=[[
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[0.95, 0.95],
                    translation_std=[0, 0, 0]),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1.0, 1.0],
                    translation_std=[0, 0, 0]),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1.05, 1.05],
                    translation_std=[0, 0, 0])
            ], [dict(type='Pack3DDetInputs', keys=['points'])]])
        tta_module = TRANSFORMS.build(tta_transform)

        results = dict()
        points = LiDARPoints(np.random.random((100, 4)), 4)
        results['points'] = points

        tta_results = tta_module(results.copy())
        assert [
            data_sample.metainfo['pcd_rotation_angle']
            for data_sample in tta_results['data_samples']
        ] == [0, 0, 0]
        assert [
            data_sample.metainfo['pcd_scale_factor']
            for data_sample in tta_results['data_samples']
        ] == [0.95, 1, 1.05]

        tta_transform = dict(
            type='TestTimeAug',
            transforms=[
                [
                    dict(
                        type='RandomFlip3D',
                        flip_ratio_bev_horizontal=0.0,
                        flip_ratio_bev_vertical=0.0),
                    dict(
                        type='RandomFlip3D',
                        flip_ratio_bev_horizontal=0.0,
                        flip_ratio_bev_vertical=1.0),
                    dict(
                        type='RandomFlip3D',
                        flip_ratio_bev_horizontal=1.0,
                        flip_ratio_bev_vertical=0.0),
                    dict(
                        type='RandomFlip3D',
                        flip_ratio_bev_horizontal=1.0,
                        flip_ratio_bev_vertical=1.0)
                ],
                [
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[pcd_rotate_range, pcd_rotate_range],
                        scale_ratio_range=[pcd_scale_factor, pcd_scale_factor],
                        translation_std=[0, 0, 0])
                    for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                    for pcd_scale_factor in [0.95, 1.0, 1.05]
                ], [dict(type='Pack3DDetInputs', keys=['points'])]
            ])
        tta_module = TRANSFORMS.build(tta_transform)

        results = dict()
        points = LiDARPoints(np.random.random((100, 4)), 4)
        results['points'] = points

        tta_results = tta_module(results.copy())
        assert [
            data_sample.metainfo['pcd_horizontal_flip']
            for data_sample in tta_results['data_samples']
        ] == [
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True
        ]
        assert [
            data_sample.metainfo['pcd_vertical_flip']
            for data_sample in tta_results['data_samples']
        ] == [
            False, False, False, False, False, False, False, False, False,
            True, True, True, True, True, True, True, True, True, False, False,
            False, False, False, False, False, False, False, True, True, True,
            True, True, True, True, True, True
        ]
        assert [
            data_sample.metainfo['pcd_rotation_angle']
            for data_sample in tta_results['data_samples']
        ] == [
            -0.78539816, -0.78539816, -0.78539816, 0.0, 0.0, 0.0, 0.78539816,
            0.78539816, 0.78539816, -0.78539816, -0.78539816, -0.78539816, 0.0,
            0.0, 0.0, 0.78539816, 0.78539816, 0.78539816, -0.78539816,
            -0.78539816, -0.78539816, 0.0, 0.0, 0.0, 0.78539816, 0.78539816,
            0.78539816, -0.78539816, -0.78539816, -0.78539816, 0.0, 0.0, 0.0,
            0.78539816, 0.78539816, 0.78539816
        ]
        assert [
            data_sample.metainfo['pcd_scale_factor']
            for data_sample in tta_results['data_samples']
        ] == [
            0.95, 1.0, 1.05, 0.95, 1.0, 1.05, 0.95, 1.0, 1.05, 0.95, 1.0, 1.05,
            0.95, 1.0, 1.05, 0.95, 1.0, 1.05, 0.95, 1.0, 1.05, 0.95, 1.0, 1.05,
            0.95, 1.0, 1.05, 0.95, 1.0, 1.05, 0.95, 1.0, 1.05, 0.95, 1.0, 1.05
        ]
