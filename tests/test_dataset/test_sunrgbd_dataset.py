import numpy as np
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
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(type='LoadAnnotations3D'),
        dict(type='IndoorFlipData', flip_ratio_yz=1.0),
        dict(
            type='IndoorGlobalRotScale',
            shift_height=True,
            rot_range=[-1 / 6, 1 / 6],
            scale_range=[0.85, 1.15]),
        dict(type='IndoorPointSample', num_points=5),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
            meta_keys=[
                'file_name', 'flip_xz', 'flip_yz', 'sample_idx', 'scale_ratio',
                'rot_angle'
            ]),
    ]

    sunrgbd_dataset = SUNRGBDDataset(root_path, ann_file, pipelines)
    data = sunrgbd_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels_3d = data['gt_labels_3d']._data
    file_name = data['img_meta']._data['file_name']
    flip_xz = data['img_meta']._data['flip_xz']
    flip_yz = data['img_meta']._data['flip_yz']
    scale_ratio = data['img_meta']._data['scale_ratio']
    rot_angle = data['img_meta']._data['rot_angle']
    sample_idx = data['img_meta']._data['sample_idx']
    assert file_name == './tests/data/sunrgbd' \
                        '/points/000001.bin'
    assert flip_xz is False
    assert flip_yz is True
    assert abs(scale_ratio - 1.0308290128214932) < 1e-5
    assert abs(rot_angle - 0.22534577750874518) < 1e-5
    assert sample_idx == 1
    expected_points = np.array([[0.6512, 1.5781, 0.0710, 0.0499],
                                [0.6473, 1.5701, 0.0657, 0.0447],
                                [0.6464, 1.5635, 0.0826, 0.0616],
                                [0.6453, 1.5603, 0.0849, 0.0638],
                                [0.6488, 1.5786, 0.0461, 0.0251]])
    expected_gt_bboxes_3d = torch.tensor(
        [[-2.0125, 3.9473, -1.2696, 2.3730, 1.9458, 2.0303, 1.2206],
         [-3.7037, 4.2396, -1.3126, 0.6032, 0.9104, 1.0033, 1.2663],
         [0.6529, 2.1638, -1.2370, 0.7348, 1.6113, 2.1694, 2.8140]])
    expected_gt_labels = np.array([0, 7, 6])
    original_classes = sunrgbd_dataset.CLASSES

    assert np.allclose(points, expected_points, 1e-2)
    assert np.allclose(gt_bboxes_3d.tensor, expected_gt_bboxes_3d, 1e-3)
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
