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
            'data_root': 'data/kitti/',
            'info_path': 'data/kitti/kitti_dbinfos_train.pkl',
            'rate': 1.0,
            'prepare': {
                'filter_by_difficulty': [-1],
                'filter_by_min_points': {
                    'Car': 5,
                    'Pedestrian': 10,
                    'Cyclist': 10
                }
            },
            'classes': ['Pedestrian', 'Cyclist', 'Car'],
            'sample_groups': {
                'Car': 12,
                'Pedestrian': 6,
                'Cyclist': 6
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
        [[6.4040, -5.7691, -2.0092, 0.4655, 1.1639, 1.8331, 0.9579],
         [10.1644, -8.0213, -1.6424, 1.6003, 3.5789, 1.5130, -0.9755],
         [24.3313, -4.8531, -1.9694, 1.6585, 4.5100, 1.4354, -1.4362],
         [18.7796, -4.2590, -1.2208, 1.5421, 3.0745, 1.3481, -1.4634],
         [35.1672, -19.8308, -2.7644, 1.5809, 3.7826, 1.6294, 1.7973],
         [11.3587, -30.8255, -1.8065, 1.5227, 3.4140, 1.4936, -0.4266],
         [15.9005, 1.2065, -1.6531, 1.5227, 3.8892, 1.2996, 2.4834],
         [13.0163, -2.0909, -1.6079, 1.5906, 3.8020, 1.3287, 1.9874],
         [18.1818, -15.5120, -1.8697, 1.6391, 4.3160, 1.3966, -0.6799],
         [13.8845, -34.5944, -1.4780, 1.4645, 3.3849, 1.3481, -2.2629],
         [14.8521, -12.5418, -1.8618, 1.6100, 3.1036, 1.5615, -1.1309],
         [6.7629, -9.0191, -1.3353, 1.5421, 3.6371, 1.5227, -0.2788],
         [26.8742, -14.1551, -1.8281, 1.5712, 3.3655, 1.3869, -1.0202],
         [10.6693, -6.4011, -1.0572, 0.8535, 0.9699, 1.8331, 0.6520],
         [10.8894, -19.2172, -2.5336, 0.5431, 1.1930, 1.7846, -1.0150],
         [30.5394, -27.5578, -1.4118, 0.6692, 0.6595, 1.7652, -0.3296],
         [25.6637, -11.1243, -1.1241, 0.6498, 1.1445, 1.8234, -1.7837],
         [12.9193, -8.4363, -1.8606, 0.4364, 0.7274, 1.6294, 1.7666],
         [19.0202, -12.1555, -1.7813, 0.6304, 1.7361, 1.6973, -2.3843],
         [3.2207, -6.5145, -2.2123, 0.5431, 1.7361, 1.7458, 0.1253],
         [18.9817, -19.5614, -1.4783, 0.6304, 1.7361, 1.6973, -1.5922],
         [13.9382, -22.5663, -0.9663, 0.4268, 1.6488, 1.7264, -1.3060]])
    expected_gt_labels_3d = torch.tensor(
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert points.shape == (5207, 4)
    assert torch.allclose(
        gt_bboxes_3d.tensor, expected_gt_bboxes_3d, atol=1e-4)
    assert torch.all(gt_labels_3d == expected_gt_labels_3d)
