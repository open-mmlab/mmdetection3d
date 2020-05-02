import numpy as np

from mmdet3d.datasets.pipelines.indoor_augment import IndoorAugment


def test_indoor_augment():
    sunrgbd_augment = IndoorAugment(0, True, True, False, False, True, True)
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
        [[-1.87572197, 3.4384955, 0.2033771, 1.66952557],
         [0.15494677, 1.15088388, -0.77230157, 0.69384689]])
    expected_sunrgbd_gt_boxes = np.array([[
        -0.45341026, 0.99208554, -1.01260705, 0.63438248, 0.59022589,
        0.89963334, -0.15403838
    ],
                                          [
                                              0.13067981, 1.50574494,
                                              -1.05946338, 1.54722899,
                                              1.68777428, 0.65598247,
                                              4.49867047
                                          ]])
    assert np.allclose(sunrgbd_point_cloud, expected_sunrgbd_point_cloud)
    assert np.allclose(sunrgbd_gt_boxes, expected_sunrgbd_gt_boxes)

    scannet_augment = IndoorAugment(
        0, True, True, True, False, False, False, rot_range=1 / 18)
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
        [[-1.61379665, 0.14011924, 0.5811581, 0.5989725],
         [-1.39004371, -0.44535946, 0.38729519, 0.40510958]])
    expected_scannet_gt_boxes = np.array([[
        -0.55030367, -0.49196554, 0.65688646, 0.66436803, 0.61192608, 0.5163464
    ], [
        0.06281816, -1.70307376, 0.60348618, 0.66448129, 0.73241497, 0.64667457
    ]])
    assert np.allclose(scannet_point_cloud, expected_scannet_point_cloud)
    assert np.allclose(scannet_gt_boxes, expected_scannet_gt_boxes)
