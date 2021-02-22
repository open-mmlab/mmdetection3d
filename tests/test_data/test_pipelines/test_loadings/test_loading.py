import mmcv
import numpy as np
import pytest
from os import path as osp

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.core.points import LiDARPoints
from mmdet3d.datasets.pipelines import (LoadAnnotations3D, LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps)


def test_load_points_from_indoor_file():
    sunrgbd_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')
    sunrgbd_load_points_from_file = LoadPointsFromFile(
        coord_type='DEPTH', load_dim=6, shift_height=True)
    sunrgbd_results = dict()
    data_path = './tests/data/sunrgbd'
    sunrgbd_info = sunrgbd_info[0]
    sunrgbd_results['pts_filename'] = osp.join(data_path,
                                               sunrgbd_info['pts_path'])
    sunrgbd_results = sunrgbd_load_points_from_file(sunrgbd_results)
    sunrgbd_point_cloud = sunrgbd_results['points'].tensor.numpy()
    assert sunrgbd_point_cloud.shape == (100, 4)

    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')
    scannet_load_data = LoadPointsFromFile(
        coord_type='DEPTH', shift_height=True)
    scannet_results = dict()
    data_path = './tests/data/scannet'
    scannet_info = scannet_info[0]

    scannet_results['pts_filename'] = osp.join(data_path,
                                               scannet_info['pts_path'])
    scannet_results = scannet_load_data(scannet_results)
    scannet_point_cloud = scannet_results['points'].tensor.numpy()
    repr_str = repr(scannet_load_data)
    expected_repr_str = 'LoadPointsFromFile(shift_height=True, ' \
                        'file_client_args={\'backend\': \'disk\'}), ' \
                        'load_dim=6, use_dim=[0, 1, 2])'
    assert repr_str == expected_repr_str
    assert scannet_point_cloud.shape == (100, 4)


def test_load_points_from_outdoor_file():
    data_path = 'tests/data/kitti/a.bin'
    load_points_from_file = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=4)
    results = dict()
    results['pts_filename'] = data_path
    results = load_points_from_file(results)
    points = results['points'].tensor.numpy()
    assert points.shape == (50, 4)
    assert np.allclose(points.sum(), 2637.479)

    load_points_from_file = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=[0, 1, 2, 3])
    results = dict()
    results['pts_filename'] = data_path
    results = load_points_from_file(results)
    new_points = results['points'].tensor.numpy()
    assert new_points.shape == (50, 4)
    assert np.allclose(points.sum(), 2637.479)
    np.equal(points, new_points)

    with pytest.raises(AssertionError):
        LoadPointsFromFile(coord_type='LIDAR', load_dim=4, use_dim=5)


def test_load_annotations3D():
    # Test scannet LoadAnnotations3D
    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')[0]
    scannet_load_annotations3D = LoadAnnotations3D(
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True)
    scannet_results = dict()
    data_path = './tests/data/scannet'

    if scannet_info['annos']['gt_num'] != 0:
        scannet_gt_bboxes_3d = scannet_info['annos']['gt_boxes_upright_depth']
        scannet_gt_labels_3d = scannet_info['annos']['class']
    else:
        scannet_gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        scannet_gt_labels_3d = np.zeros((1, ))

    # prepare input of loading pipeline
    scannet_results['ann_info'] = dict()
    scannet_results['ann_info']['pts_instance_mask_path'] = osp.join(
        data_path, scannet_info['pts_instance_mask_path'])
    scannet_results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, scannet_info['pts_semantic_mask_path'])
    scannet_results['ann_info']['gt_bboxes_3d'] = DepthInstance3DBoxes(
        scannet_gt_bboxes_3d, box_dim=6, with_yaw=False)
    scannet_results['ann_info']['gt_labels_3d'] = scannet_gt_labels_3d

    scannet_results['bbox3d_fields'] = []
    scannet_results['pts_mask_fields'] = []
    scannet_results['pts_seg_fields'] = []

    scannet_results = scannet_load_annotations3D(scannet_results)
    scannet_gt_boxes = scannet_results['gt_bboxes_3d']
    scannet_gt_lbaels = scannet_results['gt_labels_3d']

    scannet_pts_instance_mask = scannet_results['pts_instance_mask']
    scannet_pts_semantic_mask = scannet_results['pts_semantic_mask']
    repr_str = repr(scannet_load_annotations3D)
    expected_repr_str = 'LoadAnnotations3D(\n    with_bbox_3d=True,     ' \
                        'with_label_3d=True,     with_mask_3d=True,     ' \
                        'with_seg_3d=True,     with_bbox=False,     ' \
                        'with_label=False,     with_mask=False,     ' \
                        'with_seg=False,     poly2mask=True)'
    assert repr_str == expected_repr_str
    assert scannet_gt_boxes.tensor.shape == (27, 7)
    assert scannet_gt_lbaels.shape == (27, )
    assert scannet_pts_instance_mask.shape == (100, )
    assert scannet_pts_semantic_mask.shape == (100, )


def test_load_points_from_multi_sweeps():
    load_points_from_multi_sweeps = LoadPointsFromMultiSweeps()
    sweep = dict(
        data_path='./tests/data/nuscenes/sweeps/LIDAR_TOP/'
        'n008-2018-09-18-12-07-26-0400__LIDAR_TOP__1537287083900561.pcd.bin',
        timestamp=1537290014899034,
        sensor2lidar_translation=[-0.02344713, -3.88266051, -0.17151584],
        sensor2lidar_rotation=np.array(
            [[9.99979347e-01, 3.99870769e-04, 6.41441690e-03],
             [-4.42034222e-04, 9.99978299e-01, 6.57316197e-03],
             [-6.41164929e-03, -6.57586161e-03, 9.99957824e-01]]))
    points = LiDARPoints(
        np.array([[1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.],
                  [1., 2., 3., 4., 5.]]),
        points_dim=5)
    results = dict(points=points, timestamp=1537290014899034, sweeps=[sweep])

    results = load_points_from_multi_sweeps(results)
    points = results['points'].tensor.numpy()
    repr_str = repr(load_points_from_multi_sweeps)
    expected_repr_str = 'LoadPointsFromMultiSweeps(sweeps_num=10)'
    assert repr_str == expected_repr_str
    assert points.shape == (403, 4)
