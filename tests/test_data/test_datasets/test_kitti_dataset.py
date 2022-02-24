# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile

import numpy as np
import pytest
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes, limit_period
from mmdet3d.datasets import KittiDataset


def _generate_kitti_dataset_config():
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1.0, 1.0],
                    translation_std=[0, 0, 0]),
                dict(type='RandomFlip3D'),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=classes,
                    with_label=False),
                dict(type='Collect3D', keys=['points'])
            ])
    ]
    modality = dict(use_lidar=True, use_camera=False)
    split = 'training'
    return data_root, ann_file, classes, pts_prefix, pipeline, modality, split


def _generate_kitti_multi_modality_dataset_config():
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='Resize', multiscale_mode='value', keep_ratio=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1., 1.],
                    translation_std=[0, 0, 0]),
                dict(type='RandomFlip3D'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=classes,
                    with_label=False),
                dict(type='Collect3D', keys=['points', 'img'])
            ])
    ]
    modality = dict(use_lidar=True, use_camera=True)
    split = 'training'
    return data_root, ann_file, classes, pts_prefix, pipeline, modality, split


def test_getitem():
    np.random.seed(0)
    data_root, ann_file, classes, pts_prefix, \
        _, modality, split = _generate_kitti_dataset_config()
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            file_client_args=dict(backend='disk')),
        dict(
            type='ObjectSample',
            db_sampler=dict(
                data_root='tests/data/kitti/',
                # in coordinate system refactor, this test file is modified
                info_path='tests/data/kitti/kitti_dbinfos_train.pkl',
                rate=1.0,
                prepare=dict(
                    filter_by_difficulty=[-1],
                    filter_by_min_points=dict(Pedestrian=10)),
                classes=['Pedestrian', 'Cyclist', 'Car'],
                sample_groups=dict(Pedestrian=6))),
        dict(
            type='ObjectNoise',
            num_try=100,
            translation_std=[1.0, 1.0, 0.5],
            global_rot_range=[0.0, 0.0],
            rot_range=[-0.78539816, 0.78539816]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        dict(
            type='PointsRangeFilter',
            point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
        dict(
            type='ObjectRangeFilter',
            point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
        dict(type='PointShuffle'),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car']),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    data = kitti_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels_3d = data['gt_labels_3d']._data
    expected_gt_bboxes_3d = torch.tensor(
        [[9.5081, -5.2269, -1.1370, 1.2288, 0.4915, 1.9353, 1.9988]])
    expected_gt_labels_3d = torch.tensor([0])
    rot_matrix = data['img_metas']._data['pcd_rotation']
    rot_angle = data['img_metas']._data['pcd_rotation_angle']
    horizontal_flip = data['img_metas']._data['pcd_horizontal_flip']
    vertical_flip = data['img_metas']._data['pcd_vertical_flip']
    expected_rot_matrix = torch.tensor([[0.8018, 0.5976, 0.0000],
                                        [-0.5976, 0.8018, 0.0000],
                                        [0.0000, 0.0000, 1.0000]])
    expected_rot_angle = 0.6404654291602163
    noise_angle = 0.20247319
    assert torch.allclose(expected_rot_matrix, rot_matrix, atol=1e-4)
    assert math.isclose(expected_rot_angle, rot_angle, abs_tol=1e-4)
    assert horizontal_flip is True
    assert vertical_flip is False

    # after coord system refactor
    expected_gt_bboxes_3d[:, :3] = \
        expected_gt_bboxes_3d[:, :3] @ rot_matrix @ rot_matrix
    expected_gt_bboxes_3d[:, -1:] = -np.pi - expected_gt_bboxes_3d[:, -1:] \
        + 2 * rot_angle - 2 * noise_angle
    expected_gt_bboxes_3d[:, -1:] = limit_period(
        expected_gt_bboxes_3d[:, -1:], period=np.pi * 2)
    assert points.shape == (780, 4)
    assert torch.allclose(
        gt_bboxes_3d.tensor, expected_gt_bboxes_3d, atol=1e-4)
    assert torch.all(gt_labels_3d == expected_gt_labels_3d)

    # test multi-modality KITTI dataset
    np.random.seed(0)
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    multi_modality_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4),
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='Resize',
            img_scale=[(640, 192), (2560, 768)],
            multiscale_mode='range',
            keep_ratio=True),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0.2, 0.2, 0.2]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=classes),
        dict(
            type='Collect3D',
            keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
    ]
    modality = dict(use_lidar=True, use_camera=True)
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 multi_modality_pipeline, classes, modality)
    data = kitti_dataset[0]
    img = data['img']._data
    lidar2img = data['img_metas']._data['lidar2img']

    expected_lidar2img = np.array(
        [[6.02943726e+02, -7.07913330e+02, -1.22748432e+01, -1.70942719e+02],
         [1.76777252e+02, 8.80879879e+00, -7.07936157e+02, -1.02568634e+02],
         [9.99984801e-01, -1.52826728e-03, -5.29071223e-03, -3.27567995e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    assert img.shape[:] == (3, 416, 1344)
    assert np.allclose(lidar2img, expected_lidar2img)


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    data_root, ann_file, classes, pts_prefix, \
        pipeline, modality, split = _generate_kitti_dataset_config()
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))
    labels_3d = torch.tensor([
        0,
    ])
    scores_3d = torch.tensor([0.5])
    metric = ['mAP']
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    ap_dict = kitti_dataset.evaluate([result], metric)
    assert np.isclose(ap_dict['KITTI/Overall_3D_easy'], 3.0303030303030307)
    assert np.isclose(ap_dict['KITTI/Overall_3D_moderate'], 3.0303030303030307)
    assert np.isclose(ap_dict['KITTI/Overall_3D_hard'], 3.0303030303030307)


def test_show():
    from os import path as osp

    import mmcv

    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    data_root, ann_file, classes, pts_prefix, \
        pipeline, modality, split = _generate_kitti_dataset_config()
    kitti_dataset = KittiDataset(
        data_root, ann_file, split=split, modality=modality, pipeline=pipeline)
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[46.1218, -4.6496, -0.9275, 0.5316, 1.4442, 1.7450, 1.1749],
             [33.3189, 0.1981, 0.3136, 0.5656, 1.2301, 1.7985, 1.5723],
             [46.1366, -4.6404, -0.9510, 0.5162, 1.6501, 1.7540, 1.3778],
             [33.2646, 0.2297, 0.3446, 0.5746, 1.3365, 1.7947, 1.5430],
             [58.9079, 16.6272, -1.5829, 1.5656, 3.9313, 1.4899, 1.5505]]))
    scores_3d = torch.tensor([0.1815, 0.1663, 0.5792, 0.2194, 0.2780])
    labels_3d = torch.tensor([0, 0, 1, 1, 2])
    result = dict(boxes_3d=boxes_3d, scores_3d=scores_3d, labels_3d=labels_3d)
    results = [result]
    kitti_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, '000000', '000000_points.obj')
    gt_file_path = osp.join(temp_dir, '000000', '000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '000000', '000000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

    # test show with pipeline
    eval_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4),
        dict(
            type='DefaultFormatBundle3D',
            class_names=classes,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    kitti_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, '000000', '000000_points.obj')
    gt_file_path = osp.join(temp_dir, '000000', '000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '000000', '000000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

    # test multi-modality show
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    _, _, _, _, multi_modality_pipeline, modality, _ = \
        _generate_kitti_multi_modality_dataset_config()
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 multi_modality_pipeline, classes, modality)
    kitti_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, '000000', '000000_points.obj')
    gt_file_path = osp.join(temp_dir, '000000', '000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '000000', '000000_pred.obj')
    img_file_path = osp.join(temp_dir, '000000', '000000_img.png')
    img_pred_path = osp.join(temp_dir, '000000', '000000_pred.png')
    img_gt_file = osp.join(temp_dir, '000000', '000000_gt.png')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    mmcv.check_file_exist(img_file_path)
    mmcv.check_file_exist(img_pred_path)
    mmcv.check_file_exist(img_gt_file)
    tmp_dir.cleanup()

    # test multi-modality show with pipeline
    eval_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4),
        dict(type='LoadImageFromFile'),
        dict(
            type='DefaultFormatBundle3D',
            class_names=classes,
            with_label=False),
        dict(type='Collect3D', keys=['points', 'img'])
    ]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    kitti_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, '000000', '000000_points.obj')
    gt_file_path = osp.join(temp_dir, '000000', '000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '000000', '000000_pred.obj')
    img_file_path = osp.join(temp_dir, '000000', '000000_img.png')
    img_pred_path = osp.join(temp_dir, '000000', '000000_pred.png')
    img_gt_file = osp.join(temp_dir, '000000', '000000_gt.png')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    mmcv.check_file_exist(img_file_path)
    mmcv.check_file_exist(img_pred_path)
    mmcv.check_file_exist(img_gt_file)
    tmp_dir.cleanup()


def test_format_results():
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    data_root, ann_file, classes, pts_prefix, \
        pipeline, modality, split = _generate_kitti_dataset_config()
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    # coord system refactor
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900, -1.5808]]))
    labels_3d = torch.tensor([
        0,
    ])
    scores_3d = torch.tensor([0.5])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    results = [result]
    result_files, tmp_dir = kitti_dataset.format_results(results)
    expected_name = np.array(['Pedestrian'])
    expected_truncated = np.array([0.])
    expected_occluded = np.array([0])
    # coord sys refactor
    expected_alpha = np.array(-3.3410306 + np.pi)
    expected_bbox = np.array([[710.443, 144.00221, 820.29114, 307.58667]])
    expected_dimensions = np.array([[1.2, 1.89, 0.48]])
    expected_location = np.array([[1.8399826, 1.4700007, 8.410018]])
    expected_rotation_y = np.array([0.0100])
    expected_score = np.array([0.5])
    expected_sample_idx = np.array([0])
    assert np.all(result_files[0]['name'] == expected_name)
    assert np.allclose(result_files[0]['truncated'], expected_truncated)
    assert np.all(result_files[0]['occluded'] == expected_occluded)
    assert np.allclose(result_files[0]['alpha'], expected_alpha, 1e-3)
    assert np.allclose(result_files[0]['bbox'], expected_bbox)
    assert np.allclose(result_files[0]['dimensions'], expected_dimensions)
    assert np.allclose(result_files[0]['location'], expected_location)
    assert np.allclose(result_files[0]['rotation_y'], expected_rotation_y,
                       1e-3)
    assert np.allclose(result_files[0]['score'], expected_score)
    assert np.allclose(result_files[0]['sample_idx'], expected_sample_idx)
    tmp_dir.cleanup()


def test_bbox2result_kitti():
    data_root, ann_file, classes, pts_prefix, \
        pipeline, modality, split = _generate_kitti_dataset_config()
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900, -1.5808]]))
    labels_3d = torch.tensor([
        0,
    ])
    scores_3d = torch.tensor([0.5])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    results = [result]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_kitti_result_dir = tmp_dir.name
    det_annos = kitti_dataset.bbox2result_kitti(
        results, classes, submission_prefix=temp_kitti_result_dir)
    expected_file_path = os.path.join(temp_kitti_result_dir, '000000.txt')
    expected_name = np.array(['Pedestrian'])
    expected_dimensions = np.array([1.2000, 1.8900, 0.4800])
    # coord system refactor (reverse sign)
    expected_rotation_y = 0.0100
    expected_score = np.array([0.5])
    assert np.all(det_annos[0]['name'] == expected_name)
    assert np.allclose(det_annos[0]['rotation_y'], expected_rotation_y, 1e-3)
    assert np.allclose(det_annos[0]['score'], expected_score)
    assert np.allclose(det_annos[0]['dimensions'], expected_dimensions)
    assert os.path.exists(expected_file_path)
    tmp_dir.cleanup()

    tmp_dir = tempfile.TemporaryDirectory()
    temp_kitti_result_dir = tmp_dir.name
    boxes_3d = LiDARInstance3DBoxes(torch.tensor([]))
    labels_3d = torch.tensor([])
    scores_3d = torch.tensor([])
    empty_result = dict(
        boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    results = [empty_result]
    det_annos = kitti_dataset.bbox2result_kitti(
        results, classes, submission_prefix=temp_kitti_result_dir)
    expected_file_path = os.path.join(temp_kitti_result_dir, '000000.txt')
    assert os.path.exists(expected_file_path)
    tmp_dir.cleanup()


def test_bbox2result_kitti2d():
    data_root, ann_file, classes, pts_prefix, \
        pipeline, modality, split = _generate_kitti_dataset_config()
    kitti_dataset = KittiDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    bboxes = np.array([[[46.1218, -4.6496, -0.9275, 0.5316, 0.5],
                        [33.3189, 0.1981, 0.3136, 0.5656, 0.5]],
                       [[46.1366, -4.6404, -0.9510, 0.5162, 0.5],
                        [33.2646, 0.2297, 0.3446, 0.5746, 0.5]]])
    det_annos = kitti_dataset.bbox2result_kitti2d([bboxes], classes)
    expected_name = np.array(
        ['Pedestrian', 'Pedestrian', 'Cyclist', 'Cyclist'])
    expected_bbox = np.array([[46.1218, -4.6496, -0.9275, 0.5316],
                              [33.3189, 0.1981, 0.3136, 0.5656],
                              [46.1366, -4.6404, -0.951, 0.5162],
                              [33.2646, 0.2297, 0.3446, 0.5746]])
    expected_score = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.all(det_annos[0]['name'] == expected_name)
    assert np.allclose(det_annos[0]['bbox'], expected_bbox)
    assert np.allclose(det_annos[0]['score'], expected_score)
