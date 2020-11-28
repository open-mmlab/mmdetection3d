import numpy as np
import pytest
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import KittiDataset


def test_getitem():
    np.random.seed(0)
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [{
        'type': 'LoadPointsFromFile',
        'coord_type': 'LIDAR',
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


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [{
        'type': 'LoadPointsFromFile',
        'coord_type': 'LIDAR',
        'load_dim': 4,
        'use_dim': 4,
        'file_client_args': {
            'backend': 'disk'
        }
    }, {
        'type':
        'MultiScaleFlipAug3D',
        'img_scale': (1333, 800),
        'pts_scale_ratio':
        1,
        'flip':
        False,
        'transforms': [{
            'type': 'GlobalRotScaleTrans',
            'rot_range': [0, 0],
            'scale_ratio_range': [1.0, 1.0],
            'translation_std': [0, 0, 0]
        }, {
            'type': 'RandomFlip3D'
        }, {
            'type': 'PointsRangeFilter',
            'point_cloud_range': [0, -40, -3, 70.4, 40, 1]
        }, {
            'type': 'DefaultFormatBundle3D',
            'class_names': ['Pedestrian', 'Cyclist', 'Car'],
            'with_label': False
        }, {
            'type': 'Collect3D',
            'keys': ['points']
        }]
    }]
    modality = {'use_lidar': True, 'use_camera': False}
    split = 'training'
    self = KittiDataset(
        data_root,
        ann_file,
        split,
        pts_prefix,
        pipeline,
        classes,
        modality,
    )
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))
    labels_3d = torch.tensor([
        0,
    ])
    scores_3d = torch.tensor([0.5])
    metric = ['mAP']
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    ap_dict = self.evaluate([result], metric)
    assert np.isclose(ap_dict['KITTI/Overall_3D_easy'], 3.0303030303030307)
    assert np.isclose(ap_dict['KITTI/Overall_3D_moderate'], 3.0303030303030307)
    assert np.isclose(ap_dict['KITTI/Overall_3D_hard'], 3.0303030303030307)


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
            coord_type='LIDAR',
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


def test_format_results():
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [{
        'type': 'LoadPointsFromFile',
        'coord_type': 'LIDAR',
        'load_dim': 4,
        'use_dim': 4,
        'file_client_args': {
            'backend': 'disk'
        }
    }, {
        'type':
        'MultiScaleFlipAug3D',
        'img_scale': (1333, 800),
        'pts_scale_ratio':
        1,
        'flip':
        False,
        'transforms': [{
            'type': 'GlobalRotScaleTrans',
            'rot_range': [0, 0],
            'scale_ratio_range': [1.0, 1.0],
            'translation_std': [0, 0, 0]
        }, {
            'type': 'RandomFlip3D'
        }, {
            'type': 'PointsRangeFilter',
            'point_cloud_range': [0, -40, -3, 70.4, 40, 1]
        }, {
            'type': 'DefaultFormatBundle3D',
            'class_names': ['Pedestrian', 'Cyclist', 'Car'],
            'with_label': False
        }, {
            'type': 'Collect3D',
            'keys': ['points']
        }]
    }]
    modality = {'use_lidar': True, 'use_camera': False}
    split = 'training'
    self = KittiDataset(
        data_root,
        ann_file,
        split,
        pts_prefix,
        pipeline,
        classes,
        modality,
    )
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))
    labels_3d = torch.tensor([
        0,
    ])
    scores_3d = torch.tensor([0.5])
    result = dict(boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    results = [result]
    result_files, _ = self.format_results(results)
    expected_name = np.array(['Pedestrian'])
    expected_truncated = np.array([0.])
    expected_occluded = np.array([0])
    expected_alpha = np.array([-3.3410306])
    expected_bbox = np.array([[710.443, 144.00221, 820.29114, 307.58667]])
    expected_dimensions = np.array([[1.2, 1.89, 0.48]])
    expected_location = np.array([[1.8399826, 1.4700007, 8.410018]])
    expected_rotation_y = np.array([-3.1315928])
    expected_score = np.array([0.5])
    expected_sample_idx = np.array([0])
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


def test_bbox2result_kitti2d():
    data_root = 'tests/data/kitti'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [{
        'type': 'LoadPointsFromFile',
        'coord_type': 'LIDAR',
        'load_dim': 4,
        'use_dim': 4,
        'file_client_args': {
            'backend': 'disk'
        }
    }, {
        'type':
        'MultiScaleFlipAug3D',
        'img_scale': (1333, 800),
        'pts_scale_ratio':
        1,
        'flip':
        False,
        'transforms': [{
            'type': 'GlobalRotScaleTrans',
            'rot_range': [-0.1, 0.1],
            'scale_ratio_range': [0.9, 1.1],
            'translation_std': [0, 0, 0]
        }, {
            'type': 'RandomFlip3D'
        }, {
            'type': 'PointsRangeFilter',
            'point_cloud_range': [0, -40, -3, 70.4, 40, 1]
        }, {
            'type': 'DefaultFormatBundle3D',
            'class_names': ['Pedestrian', 'Cyclist', 'Car'],
            'with_label': False
        }, {
            'type': 'Collect3D',
            'keys': ['points']
        }]
    }]
    modality = {'use_lidar': True, 'use_camera': False}
    split = 'training'
    self = KittiDataset(
        data_root,
        ann_file,
        split,
        pts_prefix,
        pipeline,
        classes,
        modality,
    )
    bboxes = np.array([[[46.1218, -4.6496, -0.9275, 0.5316, 0.5],
                        [33.3189, 0.1981, 0.3136, 0.5656, 0.5]],
                       [[46.1366, -4.6404, -0.9510, 0.5162, 0.5],
                        [33.2646, 0.2297, 0.3446, 0.5746, 0.5]]])
    det_annos = self.bbox2result_kitti2d([bboxes], classes)
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
