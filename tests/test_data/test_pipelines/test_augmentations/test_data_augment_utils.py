import mmcv
import numpy as np

from mmdet3d.datasets.pipelines.data_augment_utils import (
    noise_per_object_v3_, points_transform_)


def test_noise_per_object_v3_():
    np.random.seed(0)
    points = np.fromfile(
        './tests/data/kitti/training/velodyne_reduced/000000.bin',
        np.float32).reshape(-1, 4)
    annos = mmcv.load('./tests/data/kitti/kitti_infos_train.pkl')
    info = annos[0]
    annos = info['annos']
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1).astype(np.float32)

    noise_per_object_v3_(gt_boxes=gt_bboxes_3d, points=points)
    expected_gt_bboxes_3d = np.array(
        [[3.3430212, 2.1475432, 9.388738, 1.2, 1.89, 0.48, 0.05056486]])

    assert points.shape == (800, 4)
    assert np.allclose(gt_bboxes_3d, expected_gt_bboxes_3d)


def test_points_transform():
    points = np.array([[46.5090, 6.1140, -0.7790, 0.0000],
                       [42.9490, 6.4050, -0.7050, 0.0000],
                       [42.9010, 6.5360, -0.7050, 0.0000],
                       [46.1960, 6.0960, -1.0100, 0.0000],
                       [43.3080, 6.2680, -0.9360, 0.0000]])
    gt_boxes = np.array([[
        1.5340e+01, 8.4691e+00, -1.6855e+00, 1.6400e+00, 3.7000e+00,
        1.4900e+00, 3.1300e+00
    ],
                         [
                             1.7999e+01, 8.2386e+00, -1.5802e+00, 1.5500e+00,
                             4.0200e+00, 1.5200e+00, 3.1300e+00
                         ],
                         [
                             2.9620e+01, 8.2617e+00, -1.6185e+00, 1.7800e+00,
                             4.2500e+00, 1.9000e+00, -3.1200e+00
                         ],
                         [
                             4.8218e+01, 7.8035e+00, -1.3790e+00, 1.6400e+00,
                             3.7000e+00, 1.5200e+00, -1.0000e-02
                         ],
                         [
                             3.3079e+01, -8.4817e+00, -1.3092e+00, 4.3000e-01,
                             1.7000e+00, 1.6200e+00, -1.5700e+00
                         ]])
    point_masks = np.array([[False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False]])
    loc_transforms = np.array([[-1.8635, -0.2774, -0.1774],
                               [-1.0297, -1.0302, -0.3062],
                               [1.6680, 0.2597, 0.0551],
                               [0.2230, 0.7257, -0.0097],
                               [-0.1403, 0.8300, 0.3431]])
    rot_transforms = np.array([0.6888, -0.3858, 0.1910, -0.0044, -0.0036])
    valid_mask = np.array([True, True, True, True, True])
    points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                      rot_transforms, valid_mask)
    assert points.shape == (5, 4)
    assert gt_boxes.shape == (5, 7)
