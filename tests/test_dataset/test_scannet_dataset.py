import numpy as np
import pytest
import torch

from mmdet3d.datasets import ScanNetDataset


def test_getitem():
    np.random.seed(0)
    root_path = './tests/data/scannet/'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk',
                   'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                   'sink', 'bathtub', 'garbagebin')
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_mask_3d=True,
            with_seg_3d=True),
        dict(type='IndoorPointSample', num_points=5),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=1.0,
            flip_ratio_bev_vertical=1.0),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.087266, 0.087266],
            scale_ratio_range=[1.0, 1.0],
            shift_height=True),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
                'pts_instance_mask'
            ],
            meta_keys=['file_name', 'sample_idx', 'pcd_rotation']),
    ]

    self = ScanNetDataset(root_path, ann_file, pipelines)
    data = self[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels = data['gt_labels_3d']._data
    pts_semantic_mask = data['pts_semantic_mask']._data
    pts_instance_mask = data['pts_instance_mask']._data
    file_name = data['img_metas']._data['file_name']
    pcd_rotation = data['img_metas']._data['pcd_rotation']
    sample_idx = data['img_metas']._data['sample_idx']
    expected_rotation = np.array([[0.99654, 0.08311407, 0.],
                                  [-0.08311407, 0.99654, 0.], [0., 0., 1.]])
    assert file_name == './tests/data/scannet/points/scene0000_00.bin'
    assert np.allclose(pcd_rotation, expected_rotation, 1e-3)
    assert sample_idx == 'scene0000_00'
    expected_points = torch.tensor([[-2.7231, -2.2068, 2.3543, 2.3895],
                                    [-0.4065, -3.4857, 2.1330, 2.1682],
                                    [-1.4578, 1.3510, -0.0441, -0.0089],
                                    [2.2428, -1.1323, -0.0288, 0.0064],
                                    [0.7052, -2.9752, 1.5560, 1.5912]])
    expected_gt_bboxes_3d = torch.tensor(
        [[-1.1835, -3.6317, 1.5704, 1.7577, 0.3761, 0.5724, 0.0000],
         [-3.1832, 3.2269, 1.1911, 0.6727, 0.2251, 0.6715, 0.0000],
         [-0.9598, -2.2864, 0.0093, 0.7506, 2.5709, 1.2145, 0.0000],
         [-2.6988, -2.7354, 0.8288, 0.7680, 1.8877, 0.2870, 0.0000],
         [3.2989, 0.2885, -0.0090, 0.7600, 3.8814, 2.1603, 0.0000]])
    expected_gt_labels = np.array([
        6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0,
        0, 0, 0, 5, 5, 5
    ])
    expected_pts_semantic_mask = np.array([3, 1, 2, 2, 15])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    original_classes = self.CLASSES

    assert self.CLASSES == class_names
    assert torch.allclose(points, expected_points, 1e-2)
    assert gt_bboxes_3d.tensor[:5].shape == (5, 7)
    assert torch.allclose(gt_bboxes_3d.tensor[:5], expected_gt_bboxes_3d, 1e-2)
    assert np.all(gt_labels.numpy() == expected_gt_labels)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert np.all(pts_instance_mask.numpy() == expected_pts_instance_mask)
    assert original_classes == class_names

    self = ScanNetDataset(
        root_path, ann_file, pipeline=None, classes=['cabinet', 'bed'])
    assert self.CLASSES != original_classes
    assert self.CLASSES == ['cabinet', 'bed']

    self = ScanNetDataset(
        root_path, ann_file, pipeline=None, classes=('cabinet', 'bed'))
    assert self.CLASSES != original_classes
    assert self.CLASSES == ('cabinet', 'bed')

    # Test load classes from file
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('cabinet\nbed\n')

    self = ScanNetDataset(
        root_path, ann_file, pipeline=None, classes=tmp_file.name)
    assert self.CLASSES != original_classes
    assert self.CLASSES == ['cabinet', 'bed']


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    self = ScanNetDataset(root_path, ann_file)
    results = []
    pred_boxes = dict()
    pred_boxes['boxes_3d'] = DepthInstance3DBoxes(
        torch.tensor([[
            1.4813e+00, 3.5207e+00, 1.5704e+00, 1.7445e+00, 2.3196e-01,
            5.7235e-01, 0.0000e+00
        ],
                      [
                          2.9040e+00, -3.4803e+00, 1.1911e+00, 6.6078e-01,
                          1.7072e-01, 6.7154e-01, 0.0000e+00
                      ],
                      [
                          1.1466e+00, 2.1987e+00, 9.2576e-03, 5.4184e-01,
                          2.5346e+00, 1.2145e+00, 0.0000e+00
                      ],
                      [
                          2.9168e+00, 2.5016e+00, 8.2875e-01, 6.1697e-01,
                          1.8428e+00, 2.8697e-01, 0.0000e+00
                      ],
                      [
                          -3.3114e+00, -1.3351e-02, -8.9524e-03, 4.4082e-01,
                          3.8582e+00, 2.1603e+00, 0.0000e+00
                      ],
                      [
                          -2.0135e+00, -3.4857e+00, 9.3848e-01, 1.9911e+00,
                          2.1603e-01, 1.2767e+00, 0.0000e+00
                      ],
                      [
                          -2.1945e+00, -3.1402e+00, -3.8165e-02, 1.4801e+00,
                          6.8676e-01, 1.0586e+00, 0.0000e+00
                      ],
                      [
                          -2.7553e+00, 2.4055e+00, -2.9972e-02, 1.4764e+00,
                          1.4927e+00, 2.3380e+00, 0.0000e+00
                      ]]))
    pred_boxes['labels_3d'] = torch.tensor([6, 6, 4, 9, 11, 11])
    pred_boxes['scores_3d'] = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 0.5])
    results.append(pred_boxes)
    metric = [0.25, 0.5]
    ret_dict = self.evaluate(results, metric)
    assert abs(ret_dict['table_AP_0.25'] - 0.3333) < 0.01
    assert abs(ret_dict['window_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['counter_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['curtain_AP_0.25'] - 1.0) < 0.01
