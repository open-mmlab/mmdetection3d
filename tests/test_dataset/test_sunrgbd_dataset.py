import numpy as np
import pytest
import torch

from mmdet3d.datasets import SUNRGBDDataset


def test_getitem():
    np.random.seed(0)
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                   'dresser', 'night_stand', 'bookshelf', 'bathtub')
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(type='LoadAnnotations3D'),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=0.5,
        ),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.523599, 0.523599],
            scale_ratio_range=[0.85, 1.15],
            shift_height=True),
        dict(type='IndoorPointSample', num_points=5),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
            meta_keys=[
                'file_name', 'pcd_horizontal_flip', 'sample_idx',
                'pcd_scale_factor', 'pcd_rotation'
            ]),
    ]

    sunrgbd_dataset = SUNRGBDDataset(root_path, ann_file, pipelines)
    data = sunrgbd_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels_3d = data['gt_labels_3d']._data
    file_name = data['img_metas']._data['file_name']
    pcd_horizontal_flip = data['img_metas']._data['pcd_horizontal_flip']
    pcd_scale_factor = data['img_metas']._data['pcd_scale_factor']
    pcd_rotation = data['img_metas']._data['pcd_rotation']
    sample_idx = data['img_metas']._data['sample_idx']
    pcd_rotation_expected = np.array([[0.99889565, 0.04698427, 0.],
                                      [-0.04698427, 0.99889565, 0.],
                                      [0., 0., 1.]])
    assert file_name == './tests/data/sunrgbd/points/000001.bin'
    assert pcd_horizontal_flip is False
    assert abs(pcd_scale_factor - 0.9770964398016714) < 1e-5
    assert np.allclose(pcd_rotation, pcd_rotation_expected, 1e-3)
    assert sample_idx == 1
    expected_points = torch.tensor([[-0.9904, 1.2596, 0.1105, 0.0905],
                                    [-0.9948, 1.2758, 0.0437, 0.0238],
                                    [-0.9866, 1.2641, 0.0504, 0.0304],
                                    [-0.9915, 1.2586, 0.1265, 0.1065],
                                    [-0.9890, 1.2561, 0.1216, 0.1017]])
    expected_gt_bboxes_3d = torch.tensor(
        [[0.8308, 4.1168, -1.2035, 2.2493, 1.8444, 1.9245, 1.6486],
         [2.3002, 4.8149, -1.2442, 0.5718, 0.8629, 0.9510, 1.6030],
         [-1.1477, 1.8090, -1.1725, 0.6965, 1.5273, 2.0563, 0.0552]])
    expected_gt_labels = np.array([0, 7, 6])
    original_classes = sunrgbd_dataset.CLASSES

    assert torch.allclose(points, expected_points, 1e-2)
    assert torch.allclose(gt_bboxes_3d.tensor, expected_gt_bboxes_3d, 1e-3)
    assert np.all(gt_labels_3d.numpy() == expected_gt_labels)
    assert original_classes == class_names

    SUNRGBD_dataset = SUNRGBDDataset(
        root_path, ann_file, pipeline=None, classes=['bed', 'table'])
    assert SUNRGBD_dataset.CLASSES != original_classes
    assert SUNRGBD_dataset.CLASSES == ['bed', 'table']

    SUNRGBD_dataset = SUNRGBDDataset(
        root_path, ann_file, pipeline=None, classes=('bed', 'table'))
    assert SUNRGBD_dataset.CLASSES != original_classes
    assert SUNRGBD_dataset.CLASSES == ('bed', 'table')

    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bed\ntable\n')

    SUNRGBD_dataset = SUNRGBDDataset(
        root_path, ann_file, pipeline=None, classes=tmp_file.name)
    assert SUNRGBD_dataset.CLASSES != original_classes
    assert SUNRGBD_dataset.CLASSES == ['bed', 'table']


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    sunrgbd_dataset = SUNRGBDDataset(root_path, ann_file)
    results = []
    pred_boxes = dict()
    pred_boxes['boxes_3d'] = DepthInstance3DBoxes(
        torch.tensor(
            [[1.0473, 4.1687, -1.2317, 2.3021, 1.8876, 1.9696, 1.6956],
             [2.5831, 4.8117, -1.2733, 0.5852, 0.8832, 0.9733, 1.6500],
             [-1.0864, 1.9045, -1.2000, 0.7128, 1.5631, 2.1045, 0.1022]]))
    pred_boxes['labels_3d'] = torch.tensor([0, 7, 6])
    pred_boxes['scores_3d'] = torch.tensor([0.5, 1.0, 1.0])
    results.append(pred_boxes)
    metric = [0.25, 0.5]
    ap_dict = sunrgbd_dataset.evaluate(results, metric)
    bed_precision_25 = ap_dict['bed_AP_0.25']
    dresser_precision_25 = ap_dict['dresser_AP_0.25']
    night_stand_precision_25 = ap_dict['night_stand_AP_0.25']
    assert abs(bed_precision_25 - 1) < 0.01
    assert abs(dresser_precision_25 - 1) < 0.01
    assert abs(night_stand_precision_25 - 1) < 0.01


def test_show():
    import mmcv
    import tempfile
    from os import path as osp

    from mmdet3d.core.bbox import DepthInstance3DBoxes
    temp_dir = tempfile.mkdtemp()
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    sunrgbd_dataset = SUNRGBDDataset(root_path, ann_file)
    boxes_3d = DepthInstance3DBoxes(
        torch.tensor(
            [[1.1500, 4.2614, -1.0669, 1.3219, 2.1593, 1.0267, 1.6473],
             [-0.9583, 2.1916, -1.0881, 0.6213, 1.3022, 1.6275, -3.0720],
             [2.5697, 4.8152, -1.1157, 0.5421, 0.7019, 0.7896, 1.6712],
             [0.7283, 2.5448, -1.0356, 0.7691, 0.9056, 0.5771, 1.7121],
             [-0.9860, 3.2413, -1.2349, 0.5110, 0.9940, 1.1245, 0.3295]]))
    scores_3d = torch.tensor(
        [1.5280e-01, 1.6682e-03, 6.2811e-04, 1.2860e-03, 9.4229e-06])
    labels_3d = torch.tensor([0, 0, 0, 0, 0])
    result = dict(boxes_3d=boxes_3d, scores_3d=scores_3d, labels_3d=labels_3d)
    results = [result]
    sunrgbd_dataset.show(results, temp_dir)
    pts_file_path = osp.join(temp_dir, '000001', '000001_points.obj')
    gt_file_path = osp.join(temp_dir, '000001', '000001_gt.ply')
    pred_file_path = osp.join(temp_dir, '000001', '000001_pred.ply')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
