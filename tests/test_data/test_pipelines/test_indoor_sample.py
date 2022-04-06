# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.core.points import DepthPoints
from mmdet3d.datasets.pipelines import (IndoorPatchPointSample, PointSample,
                                        PointSegClassMapping)


def test_indoor_sample():
    np.random.seed(0)
    scannet_sample_points = PointSample(5)
    scannet_results = dict()
    scannet_points = np.array([[1.0719866, -0.7870435, 0.8408122, 0.9196809],
                               [1.103661, 0.81065744, 2.6616862, 2.7405548],
                               [1.0276475, 1.5061463, 2.6174362, 2.6963048],
                               [-0.9709588, 0.6750515, 0.93901765, 1.0178864],
                               [1.0578915, 1.1693821, 0.87503505, 0.95390373],
                               [0.05560996, -1.5688863, 1.2440368, 1.3229055],
                               [-0.15731563, -1.7735453, 2.7535574, 2.832426],
                               [1.1188195, -0.99211365, 2.5551798, 2.6340485],
                               [-0.9186557, -1.7041215, 2.0562649, 2.1351335],
                               [-1.0128691, -1.3394243, 0.040936, 0.1198047]])
    scannet_results['points'] = DepthPoints(
        scannet_points, points_dim=4, attribute_dims=dict(height=3))
    scannet_pts_instance_mask = np.array(
        [15, 12, 11, 38, 0, 18, 17, 12, 17, 0])
    scannet_results['pts_instance_mask'] = scannet_pts_instance_mask
    scannet_pts_semantic_mask = np.array([38, 1, 1, 40, 0, 40, 1, 1, 1, 0])
    scannet_results['pts_semantic_mask'] = scannet_pts_semantic_mask
    scannet_results = scannet_sample_points(scannet_results)
    scannet_points_result = scannet_results['points'].tensor.numpy()
    scannet_instance_labels_result = scannet_results['pts_instance_mask']
    scannet_semantic_labels_result = scannet_results['pts_semantic_mask']
    scannet_choices = np.array([2, 8, 4, 9, 1])
    assert np.allclose(scannet_points[scannet_choices], scannet_points_result)
    assert np.all(scannet_pts_instance_mask[scannet_choices] ==
                  scannet_instance_labels_result)
    assert np.all(scannet_pts_semantic_mask[scannet_choices] ==
                  scannet_semantic_labels_result)

    np.random.seed(0)
    sunrgbd_sample_points = PointSample(5)
    sunrgbd_results = dict()
    sunrgbd_point_cloud = np.array(
        [[-1.8135729e-01, 1.4695230e+00, -1.2780589e+00, 7.8938007e-03],
         [1.2581362e-03, 2.0561588e+00, -1.0341064e+00, 2.5184631e-01],
         [6.8236995e-01, 3.3611867e+00, -9.2599887e-01, 3.5995382e-01],
         [-2.9432583e-01, 1.8714852e+00, -9.0929651e-01, 3.7665617e-01],
         [-0.5024875, 1.8032674, -1.1403012, 0.14565146],
         [-0.520559, 1.6324949, -0.9896099, 0.2963428],
         [0.95929825, 2.9402404, -0.8746674, 0.41128528],
         [-0.74624217, 1.5244724, -0.8678476, 0.41810507],
         [0.56485355, 1.5747732, -0.804522, 0.4814307],
         [-0.0913099, 1.3673826, -1.2800645, 0.00588822]])
    sunrgbd_results['points'] = DepthPoints(
        sunrgbd_point_cloud, points_dim=4, attribute_dims=dict(height=3))
    sunrgbd_results = sunrgbd_sample_points(sunrgbd_results)
    sunrgbd_choices = np.array([2, 8, 4, 9, 1])
    sunrgbd_points_result = sunrgbd_results['points'].tensor.numpy()
    repr_str = repr(sunrgbd_sample_points)
    expected_repr_str = 'PointSample(num_points=5, ' \
                        'sample_range=None, ' \
                        'replace=False)'
    assert repr_str == expected_repr_str
    assert np.allclose(sunrgbd_point_cloud[sunrgbd_choices],
                       sunrgbd_points_result)


def test_indoor_seg_sample():
    # test the train time behavior of IndoorPatchPointSample
    np.random.seed(0)
    scannet_patch_sample_points = IndoorPatchPointSample(
        5, 1.5, ignore_index=20, use_normalized_coord=True)
    scannet_seg_class_mapping = \
        PointSegClassMapping((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16,
                              24, 28, 33, 34, 36, 39), 40)
    scannet_results = dict()
    scannet_points = np.fromfile(
        './tests/data/scannet/points/scene0000_00.bin',
        dtype=np.float32).reshape((-1, 6))
    scannet_results['points'] = DepthPoints(
        scannet_points, points_dim=6, attribute_dims=dict(color=[3, 4, 5]))

    scannet_pts_semantic_mask = np.fromfile(
        './tests/data/scannet/semantic_mask/scene0000_00.bin', dtype=np.int64)
    scannet_results['pts_semantic_mask'] = scannet_pts_semantic_mask

    scannet_results = scannet_seg_class_mapping(scannet_results)
    scannet_results = scannet_patch_sample_points(scannet_results)
    scannet_points_result = scannet_results['points']
    scannet_semantic_labels_result = scannet_results['pts_semantic_mask']

    # manually constructed sampled points
    scannet_choices = np.array([87, 34, 58, 9, 18])
    scannet_center = np.array([-2.1772466, -3.4789145, 1.242711])
    scannet_center[2] = 0.0
    scannet_coord_max = np.amax(scannet_points[:, :3], axis=0)
    scannet_input_points = np.concatenate([
        scannet_points[scannet_choices, :3] - scannet_center,
        scannet_points[scannet_choices, 3:],
        scannet_points[scannet_choices, :3] / scannet_coord_max
    ], 1)

    assert scannet_points_result.points_dim == 9
    assert scannet_points_result.attribute_dims == dict(
        color=[3, 4, 5], normalized_coord=[6, 7, 8])
    scannet_points_result = scannet_points_result.tensor.numpy()
    assert np.allclose(scannet_input_points, scannet_points_result, atol=1e-6)
    assert np.all(
        np.array([13, 13, 12, 2, 0]) == scannet_semantic_labels_result)

    repr_str = repr(scannet_patch_sample_points)
    expected_repr_str = 'IndoorPatchPointSample(num_points=5, ' \
                        'block_size=1.5, ' \
                        'ignore_index=20, ' \
                        'use_normalized_coord=True, ' \
                        'num_try=10, ' \
                        'enlarge_size=0.2, ' \
                        'min_unique_num=None, ' \
                        'eps=0.01)'
    assert repr_str == expected_repr_str

    # when enlarge_size and min_unique_num are set
    np.random.seed(0)
    scannet_patch_sample_points = IndoorPatchPointSample(
        5,
        1.0,
        ignore_index=20,
        use_normalized_coord=False,
        num_try=1000,
        enlarge_size=None,
        min_unique_num=5)
    # this patch is within [0, 1] and has 5 unique points
    # it should be selected
    scannet_points = np.random.rand(5, 6)
    scannet_points[0, :3] = np.array([0.5, 0.5, 0.5])
    # generate points smaller than `min_unique_num` in local patches
    # they won't be sampled
    for i in range(2, 11, 2):
        scannet_points = np.concatenate(
            [scannet_points, np.random.rand(4, 6) + i], axis=0)
    scannet_results = dict(
        points=DepthPoints(
            scannet_points, points_dim=6,
            attribute_dims=dict(color=[3, 4, 5])),
        pts_semantic_mask=np.random.randint(0, 20,
                                            (scannet_points.shape[0], )))
    scannet_results = scannet_patch_sample_points(scannet_results)
    scannet_points_result = scannet_results['points']

    # manually constructed sampled points
    scannet_choices = np.array([2, 4, 3, 1, 0])
    scannet_center = np.array([0.56804454, 0.92559665, 0.07103606])
    scannet_center[2] = 0.0
    scannet_input_points = np.concatenate([
        scannet_points[scannet_choices, :3] - scannet_center,
        scannet_points[scannet_choices, 3:],
    ], 1)

    assert scannet_points_result.points_dim == 6
    assert scannet_points_result.attribute_dims == dict(color=[3, 4, 5])
    scannet_points_result = scannet_points_result.tensor.numpy()
    assert np.allclose(scannet_input_points, scannet_points_result, atol=1e-6)

    # test on S3DIS dataset
    np.random.seed(0)
    s3dis_patch_sample_points = IndoorPatchPointSample(
        5, 1.0, ignore_index=None, use_normalized_coord=True)
    s3dis_results = dict()
    s3dis_points = np.fromfile(
        './tests/data/s3dis/points/Area_1_office_2.bin',
        dtype=np.float32).reshape((-1, 6))
    s3dis_results['points'] = DepthPoints(
        s3dis_points, points_dim=6, attribute_dims=dict(color=[3, 4, 5]))

    s3dis_pts_semantic_mask = np.fromfile(
        './tests/data/s3dis/semantic_mask/Area_1_office_2.bin', dtype=np.int64)
    s3dis_results['pts_semantic_mask'] = s3dis_pts_semantic_mask

    s3dis_results = s3dis_patch_sample_points(s3dis_results)
    s3dis_points_result = s3dis_results['points']
    s3dis_semantic_labels_result = s3dis_results['pts_semantic_mask']

    # manually constructed sampled points
    s3dis_choices = np.array([87, 37, 60, 18, 31])
    s3dis_center = np.array([2.691, 2.231, 3.172])
    s3dis_center[2] = 0.0
    s3dis_coord_max = np.amax(s3dis_points[:, :3], axis=0)
    s3dis_input_points = np.concatenate([
        s3dis_points[s3dis_choices, :3] - s3dis_center,
        s3dis_points[s3dis_choices,
                     3:], s3dis_points[s3dis_choices, :3] / s3dis_coord_max
    ], 1)

    assert s3dis_points_result.points_dim == 9
    assert s3dis_points_result.attribute_dims == dict(
        color=[3, 4, 5], normalized_coord=[6, 7, 8])
    s3dis_points_result = s3dis_points_result.tensor.numpy()
    assert np.allclose(s3dis_input_points, s3dis_points_result, atol=1e-6)
    assert np.all(np.array([0, 1, 0, 8, 0]) == s3dis_semantic_labels_result)
