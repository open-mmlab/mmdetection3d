import os.path as osp

import mmcv
import numpy as np
import pytest

from mmdet3d.datasets.pipelines import LoadAnnotations3D, LoadPointsFromFile


def test_load_points_from_indoor_file():
    sunrgbd_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')
    sunrgbd_load_points_from_file = LoadPointsFromFile(6, shift_height=True)
    sunrgbd_results = dict()
    data_path = './tests/data/sunrgbd'
    sunrgbd_info = sunrgbd_info[0]
    sunrgbd_results['pts_filename'] = osp.join(data_path,
                                               sunrgbd_info['pts_path'])
    sunrgbd_results = sunrgbd_load_points_from_file(sunrgbd_results)
    sunrgbd_point_cloud = sunrgbd_results['points']
    assert sunrgbd_point_cloud.shape == (100, 4)

    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')
    scannet_load_data = LoadPointsFromFile(shift_height=True)
    scannet_results = dict()
    data_path = './tests/data/scannet'
    scannet_info = scannet_info[0]

    scannet_results['pts_filename'] = osp.join(data_path,
                                               scannet_info['pts_path'])
    scannet_results = scannet_load_data(scannet_results)
    scannet_point_cloud = scannet_results['points']
    assert scannet_point_cloud.shape == (100, 4)


def test_load_points_from_outdoor_file():
    data_path = 'tests/data/kitti/a.bin'
    load_points_from_file = LoadPointsFromFile(4, 4)
    results = dict()
    results['pts_filename'] = data_path
    results = load_points_from_file(results)
    points = results['points']
    assert points.shape == (50, 4)
    assert np.allclose(points.sum(), 2637.479)

    load_points_from_file = LoadPointsFromFile(4, [0, 1, 2, 3])
    results = dict()
    results['pts_filename'] = data_path
    results = load_points_from_file(results)
    new_points = results['points']
    assert new_points.shape == (50, 4)
    assert np.allclose(points.sum(), 2637.479)
    np.equal(points, new_points)

    with pytest.raises(AssertionError):
        LoadPointsFromFile(4, 5)


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
    scannet_results['ann_info']['gt_bboxes_3d'] = scannet_gt_bboxes_3d
    scannet_results['ann_info']['gt_labels_3d'] = scannet_gt_labels_3d

    scannet_results['bbox3d_fields'] = []
    scannet_results['pts_mask_fields'] = []
    scannet_results['pts_seg_fields'] = []

    scannet_results = scannet_load_annotations3D(scannet_results)
    scannet_gt_boxes = scannet_results['gt_bboxes_3d']
    scannet_gt_lbaels = scannet_results['gt_labels_3d']

    scannet_pts_instance_mask = scannet_results['pts_instance_mask']
    scannet_pts_semantic_mask = scannet_results['pts_semantic_mask']
    assert scannet_gt_boxes.shape == (27, 6)
    assert scannet_gt_lbaels.shape == (27, )
    assert scannet_pts_instance_mask.shape == (100, )
    assert scannet_pts_semantic_mask.shape == (100, )
