import numpy as np
import torch

from mmdet3d.datasets import KittiDataset


def test_getitem():
    np.random.seed(0)
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [{
        'type': 'LoadPointsFromFile',
        'load_dim': 4,
        'use_dim': 4,
        'file_client_args': {
            'backend': 'disk'
        }
    }, {
        'type': 'LoadAnnotations3D',
        'with_bbox_3d': True,
        'with_label_3d': True,
        'file_client_args': {
            'backend': 'disk'
        }
    }, {
        'type': 'ObjectSample',
        'db_sampler': {
            'data_root': 'tests/data/kitti/',
            'info_path': 'tests/data/kitti/kitti_dbinfos_train.pkl',
            'rate': 1.0,
            'prepare': {
                'filter_by_difficulty': [-1],
                'filter_by_min_points': {
                    'Pedestrian': 10
                }
            },
            'classes': ['Pedestrian', 'Cyclist', 'Car'],
            'sample_groups': {
                'Pedestrian': 6
            }
        }
    }, {
        'type': 'ObjectNoise',
        'num_try': 100,
        'translation_std': [1.0, 1.0, 0.5],
        'global_rot_range': [0.0, 0.0],
        'rot_range': [-0.78539816, 0.78539816]
    }, {
        'type': 'RandomFlip3D',
        'flip_ratio_bev_horizontal': 0.5
    }, {
        'type': 'GlobalRotScaleTrans',
        'rot_range': [-0.78539816, 0.78539816],
        'scale_ratio_range': [0.95, 1.05]
    }, {
        'type': 'PointsRangeFilter',
        'point_cloud_range': [0, -40, -3, 70.4, 40, 1]
    }, {
        'type': 'ObjectRangeFilter',
        'point_cloud_range': [0, -40, -3, 70.4, 40, 1]
    }, {
        'type': 'PointShuffle'
    }, {
        'type': 'DefaultFormatBundle3D',
        'class_names': ['Pedestrian', 'Cyclist', 'Car']
    }, {
        'type': 'Collect3D',
        'keys': ['points', 'gt_bboxes_3d', 'gt_labels_3d']
    }]
    modality = {'use_lidar': True, 'use_camera': False}
    split = 'training'
    self = KittiDataset(data_root, ann_file, split, pts_prefix, pipeline,
                        classes, modality)
    data = self[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels_3d = data['gt_labels_3d']._data
    expected_gt_bboxes_3d = torch.tensor(
        [[9.5081, -5.2269, -1.1370, 0.4915, 1.2288, 1.9353, -2.7136]])
    expected_gt_labels_3d = torch.tensor([0])
    assert points.shape == (780, 4)
    assert torch.allclose(
        gt_bboxes_3d.tensor, expected_gt_bboxes_3d, atol=1e-4)
    assert torch.all(gt_labels_3d == expected_gt_labels_3d)


def test_show():
    import mmcv
    import tempfile
    from os import path as osp

    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    temp_dir = tempfile.mkdtemp()
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    modality = {'use_lidar': True, 'use_camera': False}
    split = 'training'
    file_client_args = dict(backend='disk')
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            load_dim=4,
            use_dim=4,
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
                    class_names=class_names,
                    with_label=False),
                dict(type='Collect3D', keys=['points'])
            ])
    ]
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
    kitti_dataset.show(results, temp_dir)
    pts_file_path = osp.join(temp_dir, '000000', '000000_points.obj')
    gt_file_path = osp.join(temp_dir, '000000', '000000_gt.ply')
    pred_file_path = osp.join(temp_dir, '000000', '000000_pred.ply')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
