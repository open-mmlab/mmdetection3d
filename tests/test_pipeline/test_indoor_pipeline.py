import os.path as osp

import mmcv
import numpy as np
import torch

from mmdet3d.core.bbox import DepthInstance3DBoxes
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
    data_path = './tests/data/scannet'
    results['pts_filename'] = osp.join(data_path, info['pts_path'])
    if info['annos']['gt_num'] != 0:
        scannet_gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
            np.float32)
        scannet_gt_labels_3d = info['annos']['class'].astype(np.long)
    else:
        scannet_gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        scannet_gt_labels_3d = np.zeros((1, ), dtype=np.long)
    results['ann_info'] = dict()
    results['ann_info']['pts_instance_mask_path'] = osp.join(
        data_path, info['pts_instance_mask_path'])
    results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, info['pts_semantic_mask_path'])
    results['ann_info']['gt_bboxes_3d'] = DepthInstance3DBoxes(
        scannet_gt_bboxes_3d, box_dim=6, with_yaw=False)
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
    expected_gt_bboxes_3d = torch.tensor(
        [[-1.5005, -3.5126, 1.8565, 1.7457, 0.2415, 0.5724, 0.0000],
         [-2.8849, 3.4962, 1.5268, 0.6617, 0.1743, 0.6715, 0.0000],
         [-1.1586, -2.1924, 0.6165, 0.5557, 2.5376, 1.2145, 0.0000],
         [-2.9305, -2.4856, 0.9722, 0.6270, 1.8462, 0.2870, 0.0000],
         [3.3115, -0.0048, 1.0712, 0.4619, 3.8605, 2.1603, 0.0000]])
    expected_gt_labels_3d = np.array([
        6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0,
        0, 0, 0, 5, 5, 5
    ])
    expected_pts_semantic_mask = np.array([3, 1, 2, 2, 15])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    assert np.allclose(points, expected_points)
    assert torch.allclose(gt_bboxes_3d.tensor[:5, :], expected_gt_bboxes_3d,
                          1e-2)
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
    data_path = './tests/data/sunrgbd'
    results['pts_filename'] = osp.join(data_path, info['pts_path'])

    if info['annos']['gt_num'] != 0:
        gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
            np.float32)
        gt_labels_3d = info['annos']['class'].astype(np.long)
    else:
        gt_bboxes_3d = np.zeros((1, 7), dtype=np.float32)
        gt_labels_3d = np.zeros((1, ), dtype=np.long)

    # prepare input of pipeline
    results['ann_info'] = dict()
    results['ann_info']['gt_bboxes_3d'] = DepthInstance3DBoxes(gt_bboxes_3d)
    results['ann_info']['gt_labels_3d'] = gt_labels_3d
    results['bbox3d_fields'] = []
    results['pts_mask_fields'] = []
    results['pts_seg_fields'] = []

    results = pipeline(results)
    points = results['points']._data
    gt_bboxes_3d = results['gt_bboxes_3d']._data
    gt_labels_3d = results['gt_labels_3d']._data
    expected_points = np.array([[0.6512, 1.5781, 0.0710, 0.0499],
                                [0.6473, 1.5701, 0.0657, 0.0447],
                                [0.6464, 1.5635, 0.0826, 0.0616],
                                [0.6453, 1.5603, 0.0849, 0.0638],
                                [0.6488, 1.5786, 0.0461, 0.0251]])
    expected_gt_bboxes_3d = torch.tensor(
        [[-2.0125, 3.9473, -0.2545, 2.3730, 1.9458, 2.0303, 1.2206],
         [-3.7037, 4.2396, -0.8109, 0.6032, 0.9104, 1.0033, 1.2663],
         [0.6529, 2.1638, -0.1523, 0.7348, 1.6113, 2.1694, 2.8140]])
    expected_gt_labels_3d = np.array([0, 7, 6])
    assert torch.allclose(gt_bboxes_3d.tensor, expected_gt_bboxes_3d, 1e-3)
    assert np.allclose(gt_labels_3d.flatten(), expected_gt_labels_3d)
    assert np.allclose(points, expected_points, 1e-2)
