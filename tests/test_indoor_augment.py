import numpy as np

from mmdet3d.datasets.pipelines.indoor_augment import (IndoorFlipData,
                                                       IndoorRotateData,
                                                       IndoorShuffleData)


def test_indoor_flip_data():
    sunrgbd_flip_data = IndoorFlipData('sunrgbd')
    sunrgbd_results = dict()
    sunrgbd_results['points'] = np.array(
        [[1.02828765e+00, 3.65790772e+00, 1.97294697e-01, 1.61959505e+00],
         [-3.95979017e-01, 1.05465031e+00, -7.49204338e-01, 6.73096001e-01]])
    sunrgbd_results['gt_boxes'] = np.array([[
        0.213684, 1.036364, -0.982323, 0.61541, 0.572574, 0.872728, 3.07028526
    ],
                                            [
                                                -0.449953, 1.395455, -1.027778,
                                                1.500956, 1.637298, 0.636364,
                                                -1.58242359
                                            ]])
    sunrgbd_results = sunrgbd_flip_data(sunrgbd_results)
    sunrgbd_points = sunrgbd_results.get('points', None)
    sunrgbd_gt_boxes = sunrgbd_results.get('gt_boxes', None)
    assert sunrgbd_points.shape == (2, 4)
    assert sunrgbd_gt_boxes.shape == (2, 7)

    scannet_flip_data = IndoorFlipData('scannet')
    scannet_results = dict()
    scannet_results['points'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['gt_boxes'] = np.array([[
        0.55903838, 0.48201692, 0.65688646, 0.65370704, 0.60029864, 0.5163464
    ], [
        -0.03226406, 1.70392646, 0.60348618, 0.65165804, 0.72084366, 0.64667457
    ]])
    scannet_results = scannet_flip_data(scannet_results)
    scannet_points = scannet_results.get('points', None)
    scannet_gt_boxes = scannet_results.get('gt_boxes', None)
    assert scannet_points.shape == (2, 4)
    assert scannet_gt_boxes.shape == (2, 6)


def test_indoor_rotate_data():
    sunrgbd_indoor_rotate_data = IndoorRotateData('sunrgbd')

    sunrgbd_results = dict()
    sunrgbd_results['points'] = np.array(
        [[1.02828765e+00, 3.65790772e+00, 1.97294697e-01, 1.61959505e+00],
         [-3.95979017e-01, 1.05465031e+00, -7.49204338e-01, 6.73096001e-01]])
    sunrgbd_results['gt_boxes'] = np.array([[
        0.213684, 1.036364, -0.982323, 0.61541, 0.572574, 0.872728, 3.07028526
    ],
                                            [
                                                -0.449953, 1.395455, -1.027778,
                                                1.500956, 1.637298, 0.636364,
                                                -1.58242359
                                            ]])
    sunrgbd_results = sunrgbd_indoor_rotate_data(sunrgbd_results)
    sunrgbd_points = sunrgbd_results.get('points', None)
    sunrgbd_gt_boxes = sunrgbd_results.get('gt_boxes', None)
    assert sunrgbd_points.shape == (2, 4)
    assert sunrgbd_gt_boxes.shape == (2, 7)

    scannet_indoor_rotate_data = IndoorRotateData('scannet')
    scannet_results = dict()
    scannet_results['points'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['gt_boxes'] = np.array([[
        0.55903838, 0.48201692, 0.65688646, 0.65370704, 0.60029864, 0.5163464
    ], [
        -0.03226406, 1.70392646, 0.60348618, 0.65165804, 0.72084366, 0.64667457
    ]])
    scannet_results = scannet_indoor_rotate_data(scannet_results)
    scannet_points = scannet_results.get('points', None)
    scannet_gt_boxes = scannet_results.get('gt_boxes', None)
    assert scannet_points.shape == (2, 4)
    assert scannet_gt_boxes.shape == (2, 6)


def test_indoor_shuffle_data():
    indoor_shuffle_data = IndoorShuffleData()
    results = dict()
    results['points'] = np.array(
        [[1.02828765e+00, 3.65790772e+00, 1.97294697e-01, 1.61959505e+00],
         [-3.95979017e-01, 1.05465031e+00, -7.49204338e-01, 6.73096001e-01]])
    results = indoor_shuffle_data(results)
    points = results.get('points')
    assert points.shape == (2, 4)
