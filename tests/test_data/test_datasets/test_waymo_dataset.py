import numpy as np
import pytest
import tempfile
import torch

from mmdet3d.datasets import WaymoDataset


def _generate_waymo_dataset_config():
    data_root = 'tests/data/waymo/kitti_format/'
    ann_file = 'tests/data/waymo/kitti_format/waymo_infos_train.pkl'
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
    return data_root, ann_file, classes, pts_prefix, pipeline, modality, \
        split, point_cloud_range, file_client_args


def test_getitem():
    np.random.seed(0)
    data_root, ann_file, classes, pts_prefix, _, modality, split, \
        point_cloud_range, file_client_args = _generate_waymo_dataset_config()
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
    data_root, ann_file, classes, pts_prefix, pipeline, modality, split, \
        point_cloud_range, file_client_args = _generate_waymo_dataset_config()
    waymo_dataset = WaymoDataset(data_root, ann_file, split, pts_prefix,
                                 pipeline, classes, modality)
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[31.4750, -4.5690, 2.1857, 2.3519, 6.0931, 3.1756, -1.2895]]))
    labels_3d = torch.tensor([
        0,
    ])
    scores_3d = torch.tensor([0.5])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    # kitti protocol
    # kitti box and waymo box are in different format
    metric = ['kitti']
    _ = waymo_dataset.evaluate([result], metric=metric)
    # assert np.isclose(ap_dict['KITTI/Overall_3D_easy'], 3.030303030)
    # assert np.isclose(ap_dict['KITTI/Overall_3D_moderate'], 3.030303030)
    # assert np.isclose(ap_dict['KITTI/Overall_3D_hard'], 3.030303030)

    # waymo protocol
    # missing gt file tests/data/waymo/waymo_format/gt.bin


def test_show():
    import mmcv
    from os import path as osp

    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    # Waymo shares show function with KITTI so I just copy it here
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    data_root, ann_file, classes, pts_prefix, pipeline, modality, split, \
        point_cloud_range, file_client_args = _generate_waymo_dataset_config()
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
    pts_file_path = osp.join(temp_dir, '0000000', '0000000_points.obj')
    gt_file_path = osp.join(temp_dir, '0000000', '0000000_gt.obj')
    pred_file_path = osp.join(temp_dir, '0000000', '0000000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()
