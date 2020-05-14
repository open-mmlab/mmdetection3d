import numpy as np
import pytest
import torch

from mmdet3d.datasets import SunrgbdBaseDataset


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

    sunrgbd_dataset = SunrgbdBaseDataset(root_path, ann_file, pipelines)
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


def test_evaluate():

    if not torch.cuda.is_available():
        pytest.skip()
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    sunrgbd_dataset = SunrgbdBaseDataset(root_path, ann_file)
    results = []
    pred_boxes = dict()
    pred_boxes['box3d_lidar'] = np.array([[
        1.047307, 4.168696, -0.246859, 2.30207, 1.887584, 1.969614, 1.69564944
    ], [
        2.583086, 4.811675, -0.786667, 0.585172, 0.883176, 0.973334, 1.64999513
    ], [
        -1.086364, 1.904545, -0.147727, 0.71281, 1.563134, 2.104546, 0.1022069
    ]])
    pred_boxes['label_preds'] = torch.Tensor([0, 7, 6]).cuda()
    pred_boxes['scores'] = torch.Tensor([0.5, 1.0, 1.0]).cuda()
    results.append([pred_boxes])
    metric = [0.25, 0.5]
    ap_dict = sunrgbd_dataset.evaluate(results, metric)
    bed_precision_25 = ap_dict['bed_AP_25']
    dresser_precision_25 = ap_dict['dresser_AP_25']
    night_stand_precision_25 = ap_dict['night_stand_AP_25']
    assert abs(bed_precision_25 - 1) < 0.01
    assert abs(dresser_precision_25 - 1) < 0.01
    assert abs(night_stand_precision_25 - 1) < 0.01
