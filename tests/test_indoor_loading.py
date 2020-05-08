import os.path as osp

import mmcv
import numpy as np

from mmdet3d.datasets.pipelines.indoor_loading import (LoadAnnotations3D,
                                                       LoadPointsFromFile)


def test_load_points_from_file():
    sunrgbd_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')
    sunrgbd_load_points_from_file = LoadPointsFromFile(True, 6)
    sunrgbd_results = dict()
    data_path = './tests/data/sunrgbd/sunrgbd_trainval'
    sunrgbd_info = sunrgbd_info[0]
    scan_name = sunrgbd_info['point_cloud']['lidar_idx']
    sunrgbd_results['pts_filename'] = osp.join(data_path, 'lidar',
                                               '%06d.npy' % scan_name)
    sunrgbd_results = sunrgbd_load_points_from_file(sunrgbd_results)
    sunrgbd_point_cloud = sunrgbd_results.get('points', None)
    assert sunrgbd_point_cloud.shape == (100, 4)

    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')
    scannet_load_data = LoadPointsFromFile(True)
    scannet_results = dict()
    data_path = './tests/data/scannet/scannet_train_instance_data'
    scannet_results['data_path'] = data_path
    scannet_info = scannet_info[0]
    scan_name = scannet_info['point_cloud']['lidar_idx']

    scannet_results['pts_filename'] = osp.join(data_path,
                                               scan_name + '_vert.npy')
    scannet_results = scannet_load_data(scannet_results)
    scannet_point_cloud = scannet_results.get('points', None)
    assert scannet_point_cloud.shape == (100, 4)


def test_load_annotations3D():
    sunrgbd_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')[0]
    sunrgbd_load_annotations3D = LoadAnnotations3D()
    sunrgbd_results = dict()
    if sunrgbd_info['annos']['gt_num'] != 0:
        sunrgbd_gt_bboxes_3d = sunrgbd_info['annos']['gt_boxes_upright_depth']
        sunrgbd_gt_labels = sunrgbd_info['annos']['class'].reshape(-1, 1)
        sunrgbd_gt_bboxes_3d_mask = np.ones_like(sunrgbd_gt_labels)
    else:
        sunrgbd_gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        sunrgbd_gt_labels = np.zeros((1, 1))
        sunrgbd_gt_bboxes_3d_mask = np.zeros((1, 1))
    sunrgbd_results['gt_bboxes_3d'] = sunrgbd_gt_bboxes_3d
    sunrgbd_results['gt_labels'] = sunrgbd_gt_labels
    sunrgbd_results['gt_bboxes_3d_mask'] = sunrgbd_gt_bboxes_3d_mask
    sunrgbd_results = sunrgbd_load_annotations3D(sunrgbd_results)
    sunrgbd_gt_boxes = sunrgbd_results.get('gt_bboxes_3d', None)
    sunrgbd_gt_lbaels = sunrgbd_results.get('gt_labels', None)
    sunrgbd_gt_boxes_mask = sunrgbd_results.get('gt_bboxes_3d_mask', None)
    assert sunrgbd_gt_boxes.shape == (3, 7)
    assert sunrgbd_gt_lbaels.shape == (3, 1)
    assert sunrgbd_gt_boxes_mask.shape == (3, 1)

    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')[0]
    scannet_load_annotations3D = LoadAnnotations3D()
    scannet_results = dict()
    data_path = './tests/data/scannet/scannet_train_instance_data'
    if scannet_info['annos']['gt_num'] != 0:
        scannet_gt_bboxes_3d = scannet_info['annos']['gt_boxes_upright_depth']
        scannet_gt_labels = scannet_info['annos']['class'].reshape(-1, 1)
        scannet_gt_bboxes_3d_mask = np.ones_like(scannet_gt_labels)
    else:
        scannet_gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        scannet_gt_labels = np.zeros((1, 1))
        scannet_gt_bboxes_3d_mask = np.zeros((1, 1))
    scan_name = scannet_info['point_cloud']['lidar_idx']
    scannet_results['ins_labelname'] = osp.join(data_path,
                                                scan_name + '_ins_label.npy')
    scannet_results['sem_labelname'] = osp.join(data_path,
                                                scan_name + '_sem_label.npy')
    scannet_results['info'] = scannet_info
    scannet_results['gt_bboxes_3d'] = scannet_gt_bboxes_3d
    scannet_results['gt_labels'] = scannet_gt_labels
    scannet_results['gt_bboxes_3d_mask'] = scannet_gt_bboxes_3d_mask
    scannet_results = scannet_load_annotations3D(scannet_results)
    scannet_gt_boxes = scannet_results.get('gt_bboxes_3d', None)
    scannet_gt_lbaels = scannet_results.get('gt_labels', None)
    scannet_gt_boxes_mask = scannet_results.get('gt_bboxes_3d_mask', None)
    scannet_pts_instance_mask = scannet_results.get('pts_instance_mask', None)
    scannet_pts_semantic_mask = scannet_results.get('pts_semantic_mask', None)
    assert scannet_gt_boxes.shape == (27, 6)
    assert scannet_gt_lbaels.shape == (27, 1)
    assert scannet_gt_boxes_mask.shape == (27, 1)
    assert scannet_pts_instance_mask.shape == (100, )
    assert scannet_pts_semantic_mask.shape == (100, )
