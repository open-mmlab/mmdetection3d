# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import numpy as np
import pytest

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.core.points import DepthPoints, LiDARPoints
# yapf: disable
from mmdet3d.datasets.pipelines import (LoadAnnotations3D,
                                        LoadImageFromFileMono3D,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        NormalizePointsColor,
                                        PointSegClassMapping)

# yapf: enable


def test_load_points_from_indoor_file():
    # test on SUN RGB-D dataset with shifted height
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

    # test on ScanNet dataset with shifted height
    scannet_results['pts_filename'] = osp.join(data_path,
                                               scannet_info['pts_path'])
    scannet_results = scannet_load_data(scannet_results)
    scannet_point_cloud = scannet_results['points'].tensor.numpy()
    repr_str = repr(scannet_load_data)
    expected_repr_str = 'LoadPointsFromFile(shift_height=True, ' \
                        'use_color=False, ' \
                        'file_client_args={\'backend\': \'disk\'}, ' \
                        'load_dim=6, use_dim=[0, 1, 2])'
    assert repr_str == expected_repr_str
    assert scannet_point_cloud.shape == (100, 4)

    # test load point cloud with both shifted height and color
    scannet_load_data = LoadPointsFromFile(
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        shift_height=True,
        use_color=True)

    scannet_results = dict()

    scannet_results['pts_filename'] = osp.join(data_path,
                                               scannet_info['pts_path'])
    scannet_results = scannet_load_data(scannet_results)
    scannet_point_cloud = scannet_results['points']
    assert scannet_point_cloud.points_dim == 7
    assert scannet_point_cloud.attribute_dims == dict(
        height=3, color=[4, 5, 6])

    scannet_point_cloud = scannet_point_cloud.tensor.numpy()
    assert scannet_point_cloud.shape == (100, 7)

    # test load point cloud on S3DIS with color
    data_path = './tests/data/s3dis'
    s3dis_info = mmcv.load('./tests/data/s3dis/s3dis_infos.pkl')
    s3dis_info = s3dis_info[0]
    s3dis_load_data = LoadPointsFromFile(
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        shift_height=False,
        use_color=True)

    s3dis_results = dict()

    s3dis_results['pts_filename'] = osp.join(data_path, s3dis_info['pts_path'])
    s3dis_results = s3dis_load_data(s3dis_results)
    s3dis_point_cloud = s3dis_results['points']
    assert s3dis_point_cloud.points_dim == 6
    assert s3dis_point_cloud.attribute_dims == dict(color=[3, 4, 5])

    s3dis_point_cloud = s3dis_point_cloud.tensor.numpy()
    assert s3dis_point_cloud.shape == (100, 6)


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
    scannet_gt_labels = scannet_results['gt_labels_3d']

    scannet_pts_instance_mask = scannet_results['pts_instance_mask']
    scannet_pts_semantic_mask = scannet_results['pts_semantic_mask']
    repr_str = repr(scannet_load_annotations3D)
    expected_repr_str = 'LoadAnnotations3D(\n    with_bbox_3d=True,     ' \
                        'with_label_3d=True,     with_attr_label=False,     ' \
                        'with_mask_3d=True,     with_seg_3d=True,     ' \
                        'with_bbox=False,     with_label=False,     ' \
                        'with_mask=False,     with_seg=False,     ' \
                        'with_bbox_depth=False,     poly2mask=True)'
    assert repr_str == expected_repr_str
    assert scannet_gt_boxes.tensor.shape == (27, 7)
    assert scannet_gt_labels.shape == (27, )
    assert scannet_pts_instance_mask.shape == (100, )
    assert scannet_pts_semantic_mask.shape == (100, )

    # Test s3dis LoadAnnotations3D
    s3dis_info = mmcv.load('./tests/data/s3dis/s3dis_infos.pkl')[0]
    s3dis_load_annotations3D = LoadAnnotations3D(
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True)
    s3dis_results = dict()
    data_path = './tests/data/s3dis'

    # prepare input of loading pipeline
    s3dis_results['ann_info'] = dict()
    s3dis_results['ann_info']['pts_instance_mask_path'] = osp.join(
        data_path, s3dis_info['pts_instance_mask_path'])
    s3dis_results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, s3dis_info['pts_semantic_mask_path'])

    s3dis_results['pts_mask_fields'] = []
    s3dis_results['pts_seg_fields'] = []

    s3dis_results = s3dis_load_annotations3D(s3dis_results)

    s3dis_pts_instance_mask = s3dis_results['pts_instance_mask']
    s3dis_pts_semantic_mask = s3dis_results['pts_semantic_mask']
    assert s3dis_pts_instance_mask.shape == (100, )
    assert s3dis_pts_semantic_mask.shape == (100, )


def test_load_segmentation_mask():
    # Test loading semantic segmentation mask on ScanNet dataset
    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')[0]
    scannet_load_annotations3D = LoadAnnotations3D(
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True)
    scannet_results = dict()
    data_path = './tests/data/scannet'

    # prepare input of loading pipeline
    scannet_results['ann_info'] = dict()
    scannet_results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, scannet_info['pts_semantic_mask_path'])
    scannet_results['pts_seg_fields'] = []

    scannet_results = scannet_load_annotations3D(scannet_results)
    scannet_pts_semantic_mask = scannet_results['pts_semantic_mask']
    assert scannet_pts_semantic_mask.shape == (100, )

    # Convert class_id to label and assign ignore_index
    scannet_seg_class_mapping = \
        PointSegClassMapping((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16,
                              24, 28, 33, 34, 36, 39), 40)
    scannet_results = scannet_seg_class_mapping(scannet_results)
    scannet_pts_semantic_mask = scannet_results['pts_semantic_mask']

    assert np.all(scannet_pts_semantic_mask == np.array([
        13, 20, 1, 2, 6, 2, 13, 1, 13, 2, 0, 20, 5, 20, 2, 0, 1, 13, 0, 0, 0,
        20, 6, 20, 13, 20, 2, 20, 20, 2, 16, 5, 13, 5, 13, 0, 20, 0, 0, 1, 7,
        20, 20, 20, 20, 20, 20, 20, 0, 1, 2, 13, 16, 1, 1, 1, 6, 2, 12, 20, 3,
        20, 20, 14, 1, 20, 2, 1, 7, 2, 0, 5, 20, 5, 20, 20, 3, 6, 5, 20, 0, 13,
        12, 2, 20, 0, 0, 13, 20, 1, 20, 5, 3, 0, 13, 1, 2, 2, 2, 1
    ]))

    # Test on S3DIS dataset
    s3dis_info = mmcv.load('./tests/data/s3dis/s3dis_infos.pkl')[0]
    s3dis_load_annotations3D = LoadAnnotations3D(
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True)
    s3dis_results = dict()
    data_path = './tests/data/s3dis'

    # prepare input of loading pipeline
    s3dis_results['ann_info'] = dict()
    s3dis_results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, s3dis_info['pts_semantic_mask_path'])
    s3dis_results['pts_seg_fields'] = []

    s3dis_results = s3dis_load_annotations3D(s3dis_results)
    s3dis_pts_semantic_mask = s3dis_results['pts_semantic_mask']
    assert s3dis_pts_semantic_mask.shape == (100, )

    # Convert class_id to label and assign ignore_index
    s3dis_seg_class_mapping = PointSegClassMapping(tuple(range(13)), 13)
    s3dis_results = s3dis_seg_class_mapping(s3dis_results)
    s3dis_pts_semantic_mask = s3dis_results['pts_semantic_mask']

    assert np.all(s3dis_pts_semantic_mask == np.array([
        2, 2, 1, 2, 2, 5, 1, 0, 1, 1, 9, 12, 3, 0, 2, 0, 2, 0, 8, 2, 0, 2, 0,
        2, 1, 7, 2, 10, 2, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 0, 4, 6, 7, 2,
        1, 2, 0, 1, 7, 0, 2, 2, 2, 0, 2, 2, 1, 12, 0, 2, 2, 2, 2, 7, 2, 2, 0,
        2, 6, 2, 12, 6, 2, 12, 2, 1, 6, 1, 2, 6, 8, 2, 10, 1, 10, 0, 6, 9, 4,
        3, 0, 0, 12, 1, 1, 5, 2, 2
    ]))


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


def test_load_image_from_file_mono_3d():
    load_image_from_file_mono_3d = LoadImageFromFileMono3D()
    filename = 'tests/data/nuscenes/samples/CAM_BACK_LEFT/' \
        'n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg'
    cam_intrinsic = np.array([[1256.74, 0.0, 792.11], [0.0, 1256.74, 492.78],
                              [0.0, 0.0, 1.0]])
    input_dict = dict(
        img_prefix=None,
        img_info=dict(filename=filename, cam_intrinsic=cam_intrinsic.copy()))
    results = load_image_from_file_mono_3d(input_dict)
    assert results['img'].shape == (900, 1600, 3)
    assert np.all(results['cam2img'] == cam_intrinsic)

    repr_str = repr(load_image_from_file_mono_3d)
    expected_repr_str = 'LoadImageFromFileMono3D(to_float32=False, ' \
        "color_type='color', channel_order='bgr', " \
        "file_client_args={'backend': 'disk'})"
    assert repr_str == expected_repr_str


def test_point_seg_class_mapping():
    # max_cat_id should larger tham max id in valid_cat_ids
    with pytest.raises(AssertionError):
        point_seg_class_mapping = PointSegClassMapping([1, 2, 5], 4)

    sem_mask = np.array([
        16, 22, 2, 3, 7, 3, 16, 2, 16, 3, 1, 0, 6, 22, 3, 1, 2, 16, 1, 1, 1,
        38, 7, 25, 16, 25, 3, 40, 38, 3, 33, 6, 16, 6, 16, 1, 38, 1, 1, 2, 8,
        0, 18, 15, 0, 0, 40, 40, 1, 2, 3, 16, 33, 2, 2, 2, 7, 3, 14, 22, 4, 22,
        15, 24, 2, 40, 3, 2, 8, 3, 1, 6, 40, 6, 0, 15, 4, 7, 6, 0, 1, 16, 14,
        3, 0, 1, 1, 16, 38, 2, 15, 6, 4, 1, 16, 2, 3, 3, 3, 2
    ])
    valid_cat_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33,
                     34, 36, 39)
    point_seg_class_mapping = PointSegClassMapping(valid_cat_ids, 40)
    input_dict = dict(pts_semantic_mask=sem_mask)
    results = point_seg_class_mapping(input_dict)
    mapped_sem_mask = results['pts_semantic_mask']
    expected_sem_mask = np.array([
        13, 20, 1, 2, 6, 2, 13, 1, 13, 2, 0, 20, 5, 20, 2, 0, 1, 13, 0, 0, 0,
        20, 6, 20, 13, 20, 2, 20, 20, 2, 16, 5, 13, 5, 13, 0, 20, 0, 0, 1, 7,
        20, 20, 20, 20, 20, 20, 20, 0, 1, 2, 13, 16, 1, 1, 1, 6, 2, 12, 20, 3,
        20, 20, 14, 1, 20, 2, 1, 7, 2, 0, 5, 20, 5, 20, 20, 3, 6, 5, 20, 0, 13,
        12, 2, 20, 0, 0, 13, 20, 1, 20, 5, 3, 0, 13, 1, 2, 2, 2, 1
    ])
    repr_str = repr(point_seg_class_mapping)
    expected_repr_str = f'PointSegClassMapping(valid_cat_ids={valid_cat_ids}'\
        ', max_cat_id=40)'

    assert np.all(mapped_sem_mask == expected_sem_mask)
    assert repr_str == expected_repr_str


def test_normalize_points_color():
    coord = np.array([[68.137, 3.358, 2.516], [67.697, 3.55, 2.501],
                      [67.649, 3.76, 2.5], [66.414, 3.901, 2.459],
                      [66.012, 4.085, 2.446], [65.834, 4.178, 2.44],
                      [65.841, 4.386, 2.44], [65.745, 4.587, 2.438],
                      [65.551, 4.78, 2.432], [65.486, 4.982, 2.43]])
    color = np.array([[131, 95, 138], [71, 185, 253], [169, 47, 41],
                      [174, 161, 88], [6, 158, 213], [6, 86, 78],
                      [118, 161, 78], [72, 195, 138], [180, 170, 32],
                      [197, 85, 27]])
    points = np.concatenate([coord, color], axis=1)
    points = DepthPoints(
        points, points_dim=6, attribute_dims=dict(color=[3, 4, 5]))
    input_dict = dict(points=points)

    color_mean = [100, 150, 200]
    points_color_normalizer = NormalizePointsColor(color_mean=color_mean)
    input_dict = points_color_normalizer(input_dict)
    points = input_dict['points']
    repr_str = repr(points_color_normalizer)
    expected_repr_str = f'NormalizePointsColor(color_mean={color_mean})'

    assert repr_str == expected_repr_str
    assert np.allclose(points.coord, coord)
    assert np.allclose(points.color,
                       (color - np.array(color_mean)[None, :]) / 255.0)
