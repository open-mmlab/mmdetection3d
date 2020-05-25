import numpy as np
import torch

from mmdet3d.datasets import SUNRGBDDataset


def test_getitem():
    np.random.seed(0)
    root_path = './tests/data/sunrgbd/sunrgbd_trainval'
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
    assert file_name == './tests/data/sunrgbd/sunrgbd_trainval' \
                        '/lidar/000001.npy'
    assert flip_xz is False
    assert flip_yz is True
    assert abs(scale_ratio - 1.0308290128214932) < 1e-5
    assert abs(rot_angle - 0.22534577750874518) < 1e-5
    assert sample_idx == 1
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
    original_classes = sunrgbd_dataset.CLASSES

    assert np.allclose(points, expected_points)
    assert np.allclose(gt_bboxes_3d, expected_gt_bboxes_3d)
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
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    sunrgbd_dataset = SUNRGBDDataset(root_path, ann_file)
    results = []
    pred_boxes = dict()
    pred_boxes['boxes_3d'] = torch.Tensor(
        [[
            4.168696, -1.047307, -1.231666, 1.887584, 2.30207, 1.969614,
            1.69564944
        ],
         [
             4.811675, -2.583086, -1.273334, 0.883176, 0.585172, 0.973334,
             1.64999513
         ], [1.904545, 1.086364, -1.2, 1.563134, 0.71281, 2.104546,
             0.1022069]])
    pred_boxes['labels_3d'] = torch.Tensor([0, 7, 6])
    pred_boxes['scores_3d'] = torch.Tensor([0.5, 1.0, 1.0])
    results.append(pred_boxes)
    metric = [0.25, 0.5]
    ap_dict = sunrgbd_dataset.evaluate(results, metric)
    bed_precision_25 = ap_dict['bed_AP_0.25']
    dresser_precision_25 = ap_dict['dresser_AP_0.25']
    night_stand_precision_25 = ap_dict['night_stand_AP_0.25']
    assert abs(bed_precision_25 - 1) < 0.01
    assert abs(dresser_precision_25 - 1) < 0.01
    assert abs(night_stand_precision_25 - 1) < 0.01
