import numpy as np

from mmdet3d.datasets.pipelines.indoor_augment import (IndoorFlipData,
                                                       IndoorGlobalRotScale)


def test_indoor_flip_data():
    sunrgbd_indoor_flip_data = IndoorFlipData(0)
    sunrgbd_results = dict()
    sunrgbd_results['point_cloud'] = np.array(
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
    sunrgbd_results = sunrgbd_indoor_flip_data(sunrgbd_results)
    sunrgbd_point_cloud = sunrgbd_results.get('point_cloud', None)
    sunrgbd_gt_boxes = sunrgbd_results.get('gt_boxes', None)

    expected_sunrgbd_point_cloud = np.array(
        [[-1.02828765, 3.65790772, 0.1972947, 1.61959505],
         [0.39597902, 1.05465031, -0.74920434, 0.673096]])
    expected_sunrgbd_gt_boxes = np.array([[
        -0.213684, 1.036364, -0.982323, 0.61541, 0.572574, 0.872728, 0.07130739
    ], [
        0.449953, 1.395455, -1.027778, 1.500956, 1.637298, 0.636364, 4.72401624
    ]])
    assert np.allclose(sunrgbd_point_cloud, expected_sunrgbd_point_cloud)
    assert np.allclose(sunrgbd_gt_boxes, expected_sunrgbd_gt_boxes)

    scannet_indoor_flip_data = IndoorFlipData(0)
    scannet_results = dict()
    scannet_results['point_cloud'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['gt_boxes'] = np.array([[
        0.55903838, 0.48201692, 0.65688646, 0.65370704, 0.60029864, 0.5163464
    ], [
        -0.03226406, 1.70392646, 0.60348618, 0.65165804, 0.72084366, 0.64667457
    ]])
    scannet_results = scannet_indoor_flip_data(scannet_results)
    scannet_point_cloud = scannet_results.get('point_cloud', None)
    scannet_gt_boxes = scannet_results.get('gt_boxes', None)
    expected_scannet_point_cloud = np.array(
        [[-1.6110241, 0.16903955, 0.5811581, 0.5989725],
         [-1.3978075, -0.42035791, 0.38729519, 0.40510958]])
    expected_scannet_gt_boxes = np.array([[
        -0.55903838, -0.48201692, 0.65688646, 0.65370704, 0.60029864, 0.5163464
    ], [
        0.03226406, -1.70392646, 0.60348618, 0.65165804, 0.72084366, 0.64667457
    ]])
    assert np.allclose(scannet_point_cloud, expected_scannet_point_cloud)
    assert np.allclose(scannet_gt_boxes, expected_scannet_gt_boxes)


def test_global_rot_scale():
    sunrgbd_augment = IndoorGlobalRotScale(0, True, False, True, True)
    sunrgbd_results = dict()
    sunrgbd_results['point_cloud'] = np.array(
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

    sunrgbd_results = sunrgbd_augment(sunrgbd_results)
    sunrgbd_point_cloud = sunrgbd_results.get('point_cloud', None)
    sunrgbd_gt_boxes = sunrgbd_results.get('gt_boxes', None)
    expected_sunrgbd_point_cloud = np.array(
        [[0.89427376, 3.94489646, 0.21003141, 1.72415094],
         [-0.47835783, 1.09972989, -0.79757058, 0.71654893]])
    expected_sunrgbd_gt_boxes = np.array([[
        0.17080999, 1.11345031, -1.04573864, 0.65513891, 0.60953755,
        0.92906854, 3.01916788
    ],
                                          [
                                              -0.55427876, 1.45912611,
                                              -1.09412807, 1.59785293,
                                              1.74299674, 0.67744563,
                                              -1.63354097
                                          ]])
    assert np.allclose(sunrgbd_point_cloud, expected_sunrgbd_point_cloud)
    assert np.allclose(sunrgbd_gt_boxes, expected_sunrgbd_gt_boxes)

    scannet_augment = IndoorGlobalRotScale(
        0, True, False, False, True, rot_range=1 / 18)
    scannet_results = dict()
    scannet_results['point_cloud'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['gt_boxes'] = np.array([[
        0.55903838, 0.48201692, 0.65688646, 0.65370704, 0.60029864, 0.5163464
    ], [
        -0.03226406, 1.70392646, 0.60348618, 0.65165804, 0.72084366, 0.64667457
    ]])
    scannet_results = scannet_augment(scannet_results)
    scannet_point_cloud = scannet_results.get('point_cloud', None)
    scannet_gt_boxes = scannet_results.get('gt_boxes', None)
    expected_scannet_point_cloud = np.array(
        [[1.61240576, -0.15530836, 0.5811581, 0.5989725],
         [1.39417555, 0.43225122, 0.38729519, 0.40510958]])
    expected_scannet_gt_boxes = np.array([[
        0.55491157, 0.48676213, 0.65688646, 0.65879754, 0.60584609, 0.5163464
    ], [
        -0.04677942, 1.70358975, 0.60348618, 0.65777559, 0.72636927, 0.64667457
    ]])
    assert np.allclose(scannet_point_cloud, expected_scannet_point_cloud)
    assert np.allclose(scannet_gt_boxes, expected_scannet_gt_boxes)
