# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import tempfile
import torch

from mmdet3d.datasets import WaymoDataset


def _generate_waymo_train_dataset_config():
    data_root = 'tests/data/waymo/kitti_format/'
    ann_file = 'tests/data/waymo/kitti_format/waymo_infos_train.pkl'
    classes = ['Car', 'Pedestrian', 'Cyclist']
    pts_prefix = 'velodyne'
    point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
    file_client_args = dict(backend='disk')
    db_sampler = dict(
        data_root=data_root,
        info_path=data_root + 'waymo_dbinfos_train.pkl',
        rate=1.0,
        prepare=dict(
            filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
        classes=classes,
        sample_groups=dict(Car=15),
        points_loader=dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=file_client_args))
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=6,
            use_dim=5,
            file_client_args=file_client_args),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            file_client_args=file_client_args),
        dict(type='ObjectSample', db_sampler=db_sampler),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=0.5,
            flip_ratio_bev_vertical=0.5),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=classes),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    modality = dict(use_lidar=True, use_camera=False)
    split = 'training'
    return data_root, ann_file, classes, pts_prefix, pipeline, modality, split


def _generate_waymo_val_dataset_config():
    data_root = 'tests/data/waymo/kitti_format/'
    ann_file = 'tests/data/waymo/kitti_format/waymo_infos_val.pkl'
    classes = ['Car', 'Pedestrian', 'Cyclist']
    pts_prefix = 'velodyne'
    point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
    file_client_args = dict(backend='disk')
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=6,
            use_dim=5,
            file_client_args=file_client_args),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1., 1.],
                    translation_std=[0, 0, 0]),
                dict(type='RandomFlip3D'),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=point_cloud_range),
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


def test_getitem():
    np.random.seed(0)
    data_root, ann_file, classes, pts_prefix, pipeline, \
        modality, split = _generate_waymo_train_dataset_config()
    waymo_dataset = WaymoDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    data = waymo_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels_3d = data['gt_labels_3d']._data
    expected_gt_bboxes_3d = torch.tensor(
        [[31.4750, -4.5690, 2.1857, 2.3519, 6.0931, 3.1756, -1.2895]])
    expected_gt_labels_3d = torch.tensor([0])
    assert points.shape == (765, 5)
    assert torch.allclose(
        gt_bboxes_3d.tensor, expected_gt_bboxes_3d, atol=1e-4)
    assert torch.all(gt_labels_3d == expected_gt_labels_3d)


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    data_root, ann_file, classes, pts_prefix, pipeline, \
        modality, split = _generate_waymo_val_dataset_config()
    waymo_dataset = WaymoDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor([[
            6.9684e+01, 3.3335e+01, 4.1465e-02, 2.0100e+00, 4.3600e+00,
            1.4600e+00, -9.0000e-02
        ]]))
    labels_3d = torch.tensor([0])
    scores_3d = torch.tensor([0.5])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)

    # kitti protocol
    metric = ['kitti']
    ap_dict = waymo_dataset.evaluate([result], metric=metric)
    assert np.isclose(ap_dict['KITTI/Overall_3D_easy'], 3.0303030303030307)
    assert np.isclose(ap_dict['KITTI/Overall_3D_moderate'], 3.0303030303030307)
    assert np.isclose(ap_dict['KITTI/Overall_3D_hard'], 3.0303030303030307)

    # waymo protocol
    metric = ['waymo']
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor([[
            6.9684e+01, 3.3335e+01, 4.1465e-02, 2.0100e+00, 4.3600e+00,
            1.4600e+00, -9.0000e-02
        ]]))
    labels_3d = torch.tensor([0])
    scores_3d = torch.tensor([0.8])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    ap_dict = waymo_dataset.evaluate([result], metric=metric)
    assert np.isclose(ap_dict['Overall/L1 mAP'], 0.3333333333333333)
    assert np.isclose(ap_dict['Overall/L2 mAP'], 0.3333333333333333)
    assert np.isclose(ap_dict['Overall/L1 mAPH'], 0.3333333333333333)
    assert np.isclose(ap_dict['Overall/L2 mAPH'], 0.3333333333333333)


def test_show():
    import mmcv
    from os import path as osp

    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    # Waymo shares show function with KITTI so I just copy it here
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    data_root, ann_file, classes, pts_prefix, pipeline, \
        modality, split = _generate_waymo_val_dataset_config()
    waymo_dataset = WaymoDataset(
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
    waymo_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, '1000000', '1000000_points.obj')
    gt_file_path = osp.join(temp_dir, '1000000', '1000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '1000000', '1000000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

    # test show with pipeline
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    eval_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=6,
            use_dim=5),
        dict(
            type='DefaultFormatBundle3D',
            class_names=classes,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ]
    waymo_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, '1000000', '1000000_points.obj')
    gt_file_path = osp.join(temp_dir, '1000000', '1000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '1000000', '1000000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()


def test_format_results():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    data_root, ann_file, classes, pts_prefix, pipeline, \
        modality, split = _generate_waymo_val_dataset_config()
    waymo_dataset = WaymoDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor([[
            6.9684e+01, 3.3335e+01, 4.1465e-02, 2.0100e+00, 4.3600e+00,
            1.4600e+00, -9.0000e-02
        ]]))
    labels_3d = torch.tensor([0])
    scores_3d = torch.tensor([0.5])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    result_files, tmp_dir = waymo_dataset.format_results([result],
                                                         data_format='waymo')
    expected_name = np.array(['Car'])
    expected_truncated = np.array([0.])
    expected_occluded = np.array([0])
    expected_alpha = np.array([0.35619745])
    expected_bbox = np.array([[0., 673.59814, 37.07779, 719.7537]])
    expected_dimensions = np.array([[4.36, 1.46, 2.01]])
    expected_location = np.array([[-33.000042, 2.4999967, 68.29972]])
    expected_rotation_y = np.array([-0.09])
    expected_score = np.array([0.5])
    expected_sample_idx = np.array([1000000])
    assert np.all(result_files[0]['name'] == expected_name)
    assert np.allclose(result_files[0]['truncated'], expected_truncated)
    assert np.all(result_files[0]['occluded'] == expected_occluded)
    assert np.allclose(result_files[0]['alpha'], expected_alpha)
    assert np.allclose(result_files[0]['bbox'], expected_bbox)
    assert np.allclose(result_files[0]['dimensions'], expected_dimensions)
    assert np.allclose(result_files[0]['location'], expected_location)
    assert np.allclose(result_files[0]['rotation_y'], expected_rotation_y)
    assert np.allclose(result_files[0]['score'], expected_score)
    assert np.allclose(result_files[0]['sample_idx'], expected_sample_idx)
    tmp_dir.cleanup()
