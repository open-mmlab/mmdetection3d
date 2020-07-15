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
        [9.5081, -5.2269, -1.1370, 0.4915, 1.2288, 1.9353, -2.7136])
    expected_gt_labels_3d = torch.tensor([0])
    assert points.shape == (230, 4)
    assert torch.allclose(
        gt_bboxes_3d.tensor, expected_gt_bboxes_3d, atol=1e-4)
    assert torch.all(gt_labels_3d == expected_gt_labels_3d)


# def test_evaluate():
#     from mmdet3d.core.bbox import LiDARInstance3DBoxes
#     data_root = 'tests/data/kitti'
#     ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
#     classes = ['Pedestrian', 'Cyclist', 'Car']
#     pts_prefix = 'velodyne_reduced'
#     pipeline = [
#         {'type': 'LoadPointsFromFile', 'load_dim': 4, 'use_dim':
#         4, 'file_client_args': {'backend': 'disk'}},
#         {'type': 'MultiScaleFlipAug3D', 'img_scale': (1333, 800),
#         'pts_scale_ratio': 1, 'flip': False, 'transforms': [
#             {'type': 'GlobalRotScaleTrans', 'rot_range': [0, 0],
#             'scale_ratio_range': [1.0, 1.0],
#              'translation_std': [0, 0, 0]}, {'type': 'RandomFlip3D'},
#             {'type': 'PointsRangeFilter', 'point_cloud_range':
#             [0, -40, -3, 70.4, 40, 1]},
#             {'type': 'DefaultFormatBundle3D', 'class_names':
#             ['Pedestrian', 'Cyclist', 'Car'], 'with_label': False},
#             {'type': 'Collect3D', 'keys': ['points']}]}]
#     modality = {'use_lidar': True, 'use_camera': False}
#     split = 'training'
#     self = KittiDataset(data_root, ann_file, split, pts_prefix,
#     pipeline, classes, modality, )
#     boxes_3d = LiDARInstance3DBoxes([[7.3341, -7.3156, -1.0855,
#     0.4693, 1.1732, 1.8477, 0.7635],
#          [5.6783, -4.5259, -1.6217, 0.4693, 1.1732, 1.8477, 0.5641]
#                                      ])
#     labels_3d = torch.tensor([0, 0, 1, 1, 2, 2])
#     scores_3d = torch.tensor([0.5, 0.5, 1., 1., 1., 1.])
#     metric = ['mAP']
#     result = dict(
#         boxes_3d = boxes_3d,
#         labels_3d = labels_3d,
#         scores_3d = scores_3d
#     )
#     ap_dict = self.evaluate([result], metric)
#     print(ap_dict)
