import numpy as np
import torch

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.datasets.pipelines import IndoorFlipData, IndoorGlobalRotScale


def test_indoor_flip_data():
    np.random.seed(0)
    sunrgbd_indoor_flip_data = IndoorFlipData(1, 1)
    sunrgbd_results = dict()
    sunrgbd_results['points'] = np.array(
        [[1.02828765e+00, 3.65790772e+00, 1.97294697e-01, 1.61959505e+00],
         [-3.95979017e-01, 1.05465031e+00, -7.49204338e-01, 6.73096001e-01]])
    sunrgbd_results['gt_bboxes_3d'] = DepthInstance3DBoxes(
        np.array([[
            0.213684, 1.036364, -0.982323, 0.61541, 0.572574, 0.872728,
            3.07028526
        ],
                  [
                      -0.449953, 1.395455, -1.027778, 1.500956, 1.637298,
                      0.636364, -1.58242359
                  ]]))
    sunrgbd_results = sunrgbd_indoor_flip_data(sunrgbd_results)
    sunrgbd_points = sunrgbd_results['points']
    sunrgbd_gt_bboxes_3d = sunrgbd_results['gt_bboxes_3d']

    expected_sunrgbd_points = np.array(
        [[-1.02828765, 3.65790772, 0.1972947, 1.61959505],
         [0.39597902, 1.05465031, -0.74920434, 0.673096]])
    expected_sunrgbd_gt_bboxes_3d = torch.tensor(
        [[-0.2137, 1.0364, -0.9823, 0.6154, 0.5726, 0.8727, 0.0713],
         [0.4500, 1.3955, -1.0278, 1.5010, 1.6373, 0.6364, 4.7240]])
    assert np.allclose(sunrgbd_points, expected_sunrgbd_points)
    assert torch.allclose(sunrgbd_gt_bboxes_3d.tensor,
                          expected_sunrgbd_gt_bboxes_3d, 1e-3)

    np.random.seed(0)
    scannet_indoor_flip_data = IndoorFlipData(1, 1)
    scannet_results = dict()
    scannet_results['points'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['gt_bboxes_3d'] = DepthInstance3DBoxes(
        np.array([[
            0.55903838, 0.48201692, 0.65688646, 0.65370704, 0.60029864,
            0.5163464
        ],
                  [
                      -0.03226406, 1.70392646, 0.60348618, 0.65165804,
                      0.72084366, 0.64667457
                  ]]),
        box_dim=6,
        with_yaw=False)
    scannet_results = scannet_indoor_flip_data(scannet_results)
    scannet_points = scannet_results['points']
    scannet_gt_bboxes_3d = scannet_results['gt_bboxes_3d']

    expected_scannet_points = np.array(
        [[-1.6110241, 0.16903955, 0.5811581, 0.5989725],
         [-1.3978075, -0.42035791, 0.38729519, 0.40510958]])
    expected_scannet_gt_bboxes_3d = torch.tensor(
        [[-0.5590, -0.4820, 0.6569, 0.6537, 0.6003, 0.5163, 0.0000],
         [0.0323, -1.7039, 0.6035, 0.6517, 0.7208, 0.6467, 0.0000]])
    assert np.allclose(scannet_points, expected_scannet_points)
    assert torch.allclose(scannet_gt_bboxes_3d.tensor,
                          expected_scannet_gt_bboxes_3d, 1e-2)


def test_global_rot_scale():
    np.random.seed(0)
    sunrgbd_augment = IndoorGlobalRotScale(
        True, rot_range=[-1 / 6, 1 / 6], scale_range=[0.85, 1.15])
    sunrgbd_results = dict()
    sunrgbd_results['points'] = np.array(
        [[1.02828765e+00, 3.65790772e+00, 1.97294697e-01, 1.61959505e+00],
         [-3.95979017e-01, 1.05465031e+00, -7.49204338e-01, 6.73096001e-01]])
    sunrgbd_results['gt_bboxes_3d'] = DepthInstance3DBoxes(
        np.array([[
            0.213684, 1.036364, -0.982323, 0.61541, 0.572574, 0.872728,
            3.07028526
        ],
                  [
                      -0.449953, 1.395455, -1.027778, 1.500956, 1.637298,
                      0.636364, -1.58242359
                  ]]))

    sunrgbd_results = sunrgbd_augment(sunrgbd_results)
    sunrgbd_points = sunrgbd_results['points']
    sunrgbd_gt_bboxes_3d = sunrgbd_results['gt_bboxes_3d']

    expected_sunrgbd_points = np.array(
        [[0.89427376, 3.94489646, 0.21003141, 1.72415094],
         [-0.47835783, 1.09972989, -0.79757058, 0.71654893]])
    expected_sunrgbd_gt_bboxes_3d = torch.tensor(
        [[0.1708, 1.1135, -1.0457, 0.6551, 0.6095, 0.9291, 3.0192],
         [-0.5543, 1.4591, -1.0941, 1.5979, 1.7430, 0.6774, -1.6335]])
    assert np.allclose(sunrgbd_points, expected_sunrgbd_points)
    assert torch.allclose(sunrgbd_gt_bboxes_3d.tensor,
                          expected_sunrgbd_gt_bboxes_3d, 1e-3)

    np.random.seed(0)
    scannet_augment = IndoorGlobalRotScale(
        True, rot_range=[-1 * 1 / 36, 1 / 36], scale_range=None)
    scannet_results = dict()
    scannet_results['points'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['gt_bboxes_3d'] = DepthInstance3DBoxes(
        np.array([[
            0.55903838, 0.48201692, 0.65688646, 0.65370704, 0.60029864,
            0.5163464
        ],
                  [
                      -0.03226406, 1.70392646, 0.60348618, 0.65165804,
                      0.72084366, 0.64667457
                  ]]),
        box_dim=6,
        with_yaw=False)
    scannet_results = scannet_augment(scannet_results)
    scannet_points = scannet_results['points']
    scannet_gt_bboxes_3d = scannet_results['gt_bboxes_3d']

    expected_scannet_points = np.array(
        [[1.61240576, -0.15530836, 0.5811581, 0.5989725],
         [1.39417555, 0.43225122, 0.38729519, 0.40510958]])
    expected_scannet_gt_bboxes_3d = torch.tensor(
        [[0.5549, 0.4868, 0.6569, 0.6588, 0.6058, 0.5163, 0.0000],
         [-0.0468, 1.7036, 0.6035, 0.6578, 0.7264, 0.6467, 0.0000]])
    assert np.allclose(scannet_points, expected_scannet_points)
    assert torch.allclose(scannet_gt_bboxes_3d.tensor,
                          expected_scannet_gt_bboxes_3d, 1e-3)
