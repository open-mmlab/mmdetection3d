import numpy as np
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
        dict(type='IndoorFlipData', flip_ratio_yz=1.0, flip_ratio_xz=1.0),
        dict(
            type='IndoorGlobalRotScale',
            shift_height=True,
            rot_range=[-1 / 36, 1 / 36],
            scale_range=None),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
                'pts_instance_mask'
            ],
            meta_keys=[
                'file_name', 'flip_xz', 'flip_yz', 'sample_idx', 'rot_angle'
            ]),
    ]

    scannet_dataset = ScanNetDataset(root_path, ann_file, pipelines)
    data = scannet_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels = data['gt_labels_3d']._data
    pts_semantic_mask = data['pts_semantic_mask']._data
    pts_instance_mask = data['pts_instance_mask']._data
    file_name = data['img_meta']._data['file_name']
    flip_xz = data['img_meta']._data['flip_xz']
    flip_yz = data['img_meta']._data['flip_yz']
    rot_angle = data['img_meta']._data['rot_angle']
    sample_idx = data['img_meta']._data['sample_idx']
    assert file_name == './tests/data/scannet/' \
                        'points/scene0000_00.bin'
    assert flip_xz is True
    assert flip_yz is True
    assert abs(rot_angle - (-0.005471397477913809)) < 1e-5
    assert sample_idx == 'scene0000_00'
    expected_points = np.array(
        [[-2.9078157, -1.9569951, 2.3543026, 2.389488],
         [-0.71360034, -3.4359822, 2.1330001, 2.1681855],
         [-1.332374, 1.474838, -0.04405887, -0.00887359],
         [2.1336637, -1.3265059, -0.02880373, 0.00638155],
         [0.43895668, -3.0259454, 1.5560012, 1.5911865]])
    expected_gt_bboxes_3d = np.array([
        [-1.5005362, -3.512584, 1.8565295, 1.7457027, 0.24149807, 0.57235193],
        [-2.8848705, 3.4961755, 1.5268247, 0.66170084, 0.17433672, 0.67153597],
        [-1.1585636, -2.192365, 0.61649567, 0.5557011, 2.5375574, 1.2144762],
        [-2.930457, -2.4856408, 0.9722377, 0.6270478, 1.8461524, 0.28697443],
        [3.3114715, -0.00476722, 1.0712197, 0.46191898, 3.8605113, 2.1603441]
    ])
    expected_gt_labels = np.array([
        6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0,
        0, 0, 0, 5, 5, 5
    ])
    expected_pts_semantic_mask = np.array([3, 1, 2, 2, 15])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    original_classes = scannet_dataset.CLASSES

    assert scannet_dataset.CLASSES == class_names
    assert np.allclose(points, expected_points)
    assert gt_bboxes_3d[:5].shape == (5, 6)
    assert np.allclose(gt_bboxes_3d[:5], expected_gt_bboxes_3d)
    assert np.all(gt_labels.numpy() == expected_gt_labels)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert np.all(pts_instance_mask.numpy() == expected_pts_instance_mask)
    assert original_classes == class_names

    scannet_dataset = ScanNetDataset(
        root_path, ann_file, pipeline=None, classes=['cabinet', 'bed'])
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ['cabinet', 'bed']

    scannet_dataset = ScanNetDataset(
        root_path, ann_file, pipeline=None, classes=('cabinet', 'bed'))
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ('cabinet', 'bed')

    # Test load classes from file
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('cabinet\nbed\n')

    scannet_dataset = ScanNetDataset(
        root_path, ann_file, pipeline=None, classes=tmp_file.name)
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ['cabinet', 'bed']


def test_evaluate():
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetDataset(root_path, ann_file)
    results = []
    pred_boxes = dict()
    pred_boxes['boxes_3d'] = torch.Tensor(
        [[
            3.52074146e+00, -1.48129511e+00, 1.57035351e+00, 2.31956959e-01,
            1.74445975e+00, 5.72351933e-01, 0
        ],
         [
             -3.48033905e+00, -2.90395617e+00, 1.19105673e+00, 1.70723915e-01,
             6.60776615e-01, 6.71535969e-01, 0
         ],
         [
             2.19867110e+00, -1.14655101e+00, 9.25755501e-03, 2.53463078e+00,
             5.41841269e-01, 1.21447623e+00, 0
         ],
         [
             2.50163722, -2.91681337, 0.82875049, 1.84280431, 0.61697435,
             0.28697443, 0
         ],
         [
             -0.01335114, 3.3114481, -0.00895238, 3.85815716, 0.44081616,
             2.16034412, 0
         ]])
    pred_boxes['labels_3d'] = torch.Tensor([6, 6, 4, 9, 11])
    pred_boxes['scores_3d'] = torch.Tensor([0.5, 1.0, 1.0, 1.0, 1.0])
    results.append(pred_boxes)
    metric = [0.25, 0.5]
    ret_dict = scannet_dataset.evaluate(results, metric)
    table_average_precision_25 = ret_dict['table_AP_0.25']
    window_average_precision_25 = ret_dict['window_AP_0.25']
    counter_average_precision_25 = ret_dict['counter_AP_0.25']
    curtain_average_precision_25 = ret_dict['curtain_AP_0.25']
    assert abs(table_average_precision_25 - 0.3333) < 0.01
    assert abs(window_average_precision_25 - 1) < 0.01
    assert abs(counter_average_precision_25 - 1) < 0.01
    assert abs(curtain_average_precision_25 - 0.5) < 0.01
