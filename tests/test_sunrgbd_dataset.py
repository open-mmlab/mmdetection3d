import numpy as np

from mmdet3d.datasets import SunrgbdDataset


def test_getitem():
    np.random.seed(0)
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                   'dresser', 'night_stand', 'bookshelf', 'bathtub')
    pipelines = [
        dict(
            type='IndoorLoadPointsFromFile',
            use_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(type='IndoorFlipData', flip_ratio_yz=1.0),
        dict(
            type='IndoorGlobalRotScale',
            use_height=True,
            rot_range=[-np.pi / 6, np.pi / 6],
            scale_range=[0.85, 1.15]),
        dict(type='IndoorPointSample', num_points=5),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels']),
    ]

    sunrgbd_dataset = SunrgbdDataset(root_path, ann_file, pipelines, True)
    data = sunrgbd_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels = data['gt_labels']._data

    expected_points = np.array(
        [[0.6570105, 1.5538014, 0.24514851, 1.0165423],
         [0.656101, 1.558591, 0.21755838, 0.98895216],
         [0.6293659, 1.5679953, -0.10004003, 0.67135376],
         [0.6068739, 1.5974995, -0.41063973, 0.36075398],
         [0.6464709, 1.5573514, 0.15114647, 0.9225402]])
    expected_gt_bboxes_3d = np.array([[
        -2.012483, 3.9473376, -0.25446942, 2.3730404, 1.9457763, 2.0303352,
        1.2205974
    ],
                                      [
                                          -3.7036808, 4.2396426, -0.81091917,
                                          0.6032123, 0.91040343, 1.003341,
                                          1.2662518
                                      ],
                                      [
                                          0.6528646, 2.1638472, -0.15228128,
                                          0.7347852, 1.6113238, 2.1694272,
                                          2.81404
                                      ]])
    expected_gt_labels = np.array([0, 7, 6])

    assert np.allclose(points, expected_points)
    assert np.allclose(gt_bboxes_3d, expected_gt_bboxes_3d)
    assert np.all(gt_labels.numpy() == expected_gt_labels)
