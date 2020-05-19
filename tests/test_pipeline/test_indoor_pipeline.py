import os.path as osp

import mmcv
import numpy as np

from mmdet3d.datasets.pipelines import Compose


def test_scannet_pipeline():
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk',
                   'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                   'sink', 'bathtub', 'garbagebin')

    np.random.seed(0)
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
            ]),
    ]
    pipeline = Compose(pipelines)
    info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')[0]
    results = dict()
    data_path = './tests/data/scannet/scannet_train_instance_data'
    results['data_path'] = data_path
    scan_name = info['point_cloud']['lidar_idx']
    results['pts_filename'] = osp.join(data_path, f'{scan_name}_vert.npy')
    if info['annos']['gt_num'] != 0:
        scannet_gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']
        scannet_gt_labels_3d = info['annos']['class']
    else:
        scannet_gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        scannet_gt_labels_3d = np.zeros((1, ))
    scan_name = info['point_cloud']['lidar_idx']
    results['ann_info'] = dict()
    results['ann_info']['pts_instance_mask_path'] = osp.join(
        data_path, f'{scan_name}_ins_label.npy')
    results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, f'{scan_name}_sem_label.npy')
    results['ann_info']['gt_bboxes_3d'] = scannet_gt_bboxes_3d
    results['ann_info']['gt_labels_3d'] = scannet_gt_labels_3d

    results['bbox3d_fields'] = []
    results['pts_mask_fields'] = []
    results['pts_seg_fields'] = []

    results = pipeline(results)

    points = results['points']._data
    gt_bboxes_3d = results['gt_bboxes_3d']._data
    gt_labels_3d = results['gt_labels_3d']._data
    pts_semantic_mask = results['pts_semantic_mask']._data
    pts_instance_mask = results['pts_instance_mask']._data
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
    expected_gt_labels_3d = np.array([
        6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0,
        0, 0, 0, 5, 5, 5
    ])
    expected_pts_semantic_mask = np.array([3, 1, 2, 2, 15])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    assert np.allclose(points, expected_points)
    assert np.allclose(gt_bboxes_3d[:5, :], expected_gt_bboxes_3d)
    assert np.all(gt_labels_3d.numpy() == expected_gt_labels_3d)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert np.all(pts_instance_mask.numpy() == expected_pts_instance_mask)


def test_sunrgbd_pipeline():
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                   'dresser', 'night_stand', 'bookshelf', 'bathtub')
    np.random.seed(0)
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
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
    ]
    pipeline = Compose(pipelines)
    results = dict()
    info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')[0]
    data_path = './tests/data/sunrgbd/sunrgbd_trainval'
    scan_name = info['point_cloud']['lidar_idx']
    results['pts_filename'] = osp.join(data_path, 'lidar',
                                       f'{scan_name:06d}.npy')

    if info['annos']['gt_num'] != 0:
        gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']
        gt_labels_3d = info['annos']['class']
    else:
        gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        gt_labels_3d = np.zeros((1, ))

    # prepare input of pipeline
    results['ann_info'] = dict()
    results['ann_info']['gt_bboxes_3d'] = gt_bboxes_3d
    results['ann_info']['gt_labels_3d'] = gt_labels_3d
    results['bbox3d_fields'] = []
    results['pts_mask_fields'] = []
    results['pts_seg_fields'] = []

    results = pipeline(results)
    points = results['points']._data
    gt_bboxes_3d = results['gt_bboxes_3d']._data
    gt_labels_3d = results['gt_labels_3d']._data
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
    expected_gt_labels_3d = np.array([0, 7, 6])
    assert np.allclose(gt_bboxes_3d, expected_gt_bboxes_3d)
    assert np.allclose(gt_labels_3d.flatten(), expected_gt_labels_3d)
    assert np.allclose(points, expected_points)
