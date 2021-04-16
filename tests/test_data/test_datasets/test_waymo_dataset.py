import numpy as np
import torch

from mmdet3d.datasets import WaymoDataset


def _generate_waymo_dataset_config():
    data_root = 'tests/data/waymo/'
    ann_file = 'tests/data/waymo/waymo_infos_train.pkl'
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
