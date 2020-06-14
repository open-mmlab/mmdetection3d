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
    expected_gt_bboxes_3d = torch.tensor(
        [[-1.5005, -3.5126, 1.5704, 1.7457, 0.2415, 0.5724, 0.0000],
         [-2.8849, 3.4962, 1.1911, 0.6617, 0.1743, 0.6715, 0.0000],
         [-1.1586, -2.1924, 0.0093, 0.5557, 2.5376, 1.2145, 0.0000],
         [-2.9305, -2.4856, 0.8288, 0.6270, 1.8462, 0.2870, 0.0000],
         [3.3115, -0.0048, -0.0090, 0.4619, 3.8605, 2.1603, 0.0000]])
    expected_gt_labels = np.array([
        6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0,
        0, 0, 0, 5, 5, 5
    ])
    expected_pts_semantic_mask = np.array([3, 1, 2, 2, 15])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    original_classes = scannet_dataset.CLASSES

    assert scannet_dataset.CLASSES == class_names
    assert np.allclose(points, expected_points)
    assert gt_bboxes_3d.tensor[:5].shape == (5, 7)
    assert torch.allclose(gt_bboxes_3d.tensor[:5], expected_gt_bboxes_3d, 1e-2)
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
    from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetDataset(root_path, ann_file)
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
    ret_dict = scannet_dataset.evaluate(results, metric)
    assert abs(ret_dict['table_AP_0.25'] - 0.3333) < 0.01
    assert abs(ret_dict['window_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['counter_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['curtain_AP_0.25'] - 1.0) < 0.01
