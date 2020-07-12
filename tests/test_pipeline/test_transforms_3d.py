import mmcv
import numpy as np
import torch

from mmdet3d.core import Box3DMode, CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.datasets import ObjectNoise, ObjectSample


def test_remove_points_in_boxes():
    points = np.array([[68.1370, 3.3580, 2.5160, 0.0000],
                       [67.6970, 3.5500, 2.5010, 0.0000],
                       [67.6490, 3.7600, 2.5000, 0.0000],
                       [66.4140, 3.9010, 2.4590, 0.0000],
                       [66.0120, 4.0850, 2.4460, 0.0000],
                       [65.8340, 4.1780, 2.4400, 0.0000],
                       [65.8410, 4.3860, 2.4400, 0.0000],
                       [65.7450, 4.5870, 2.4380, 0.0000],
                       [65.5510, 4.7800, 2.4320, 0.0000],
                       [65.4860, 4.9820, 2.4300, 0.0000]])

    boxes = np.array(
        [[30.0285, 10.5110, -1.5304, 0.5100, 0.8700, 1.6000, 1.6400],
         [7.8369, 1.6053, -1.5605, 0.5800, 1.2300, 1.8200, -3.1000],
         [10.8740, -1.0827, -1.3310, 0.6000, 0.5200, 1.7100, 1.3500],
         [14.9783, 2.2466, -1.4950, 0.6100, 0.7300, 1.5300, -1.9200],
         [11.0656, 0.6195, -1.5202, 0.6600, 1.0100, 1.7600, -1.4600],
         [10.5994, -7.9049, -1.4980, 0.5300, 1.9600, 1.6800, 1.5600],
         [28.7068, -8.8244, -1.1485, 0.6500, 1.7900, 1.7500, 3.1200],
         [20.2630, 5.1947, -1.4799, 0.7300, 1.7600, 1.7300, 1.5100],
         [18.2496, 3.1887, -1.6109, 0.5600, 1.6800, 1.7100, 1.5600],
         [7.7396, -4.3245, -1.5801, 0.5600, 1.7900, 1.8000, -0.8300]])

    points = ObjectSample.remove_points_in_boxes(points, boxes)
    assert points.shape == (10, 4)


def test_object_sample():
    import pickle
    db_sampler = mmcv.ConfigDict({
        'data_root': './tests/data/kitti/',
        'info_path': './tests/data/kitti/kitti_dbinfos_train.pkl',
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
    })
    with open('./tests/data/kitti/kitti_dbinfos_train.pkl', 'rb') as f:
        db_infos = pickle.load(f)
    np.random.seed(0)
    object_sample = ObjectSample(db_sampler)
    points = np.fromfile(
        './tests/data/kitti/training/velodyne_reduced/000000.bin',
        np.float32).reshape(-1, 4)
    annos = mmcv.load('./tests/data/kitti/kitti_infos_train.pkl')
    info = annos[0]
    annos = info['annos']
    gt_names = annos['name']
    gt_bboxes_3d = db_infos['Pedestrian'][0]['box3d_lidar']
    gt_bboxes_3d = LiDARInstance3DBoxes([gt_bboxes_3d])
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    gt_labels = []
    for cat in gt_names:
        if cat in CLASSES:
            gt_labels.append(CLASSES.index(cat))
        else:
            gt_labels.append(-1)
    input_dict = dict(
        points=points, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels)
    input_dict = object_sample(input_dict)
    points = input_dict['points']
    gt_bboxes_3d = input_dict['gt_bboxes_3d']
    gt_labels_3d = input_dict['gt_labels_3d']
    repr_str = repr(object_sample)
    expected_repr_str = 'ObjectSample'
    assert repr_str == expected_repr_str
    assert points.shape == (1177, 4)
    assert gt_bboxes_3d.tensor.shape == (2, 7)
    assert np.all(gt_labels_3d == [1, 0])


def test_object_noise():
    np.random.seed(0)
    object_noise = ObjectNoise()
    points = np.fromfile(
        './tests/data/kitti/training/velodyne_reduced/000000.bin',
        np.float32).reshape(-1, 4)
    annos = mmcv.load('./tests/data/kitti/kitti_infos_train.pkl')
    info = annos[0]
    rect = info['calib']['R0_rect'].astype(np.float32)
    Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
    annos = info['annos']
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1).astype(np.float32)
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
    input_dict = dict(points=points, gt_bboxes_3d=gt_bboxes_3d)
    input_dict = object_noise(input_dict)
    points = input_dict['points']
    gt_bboxes_3d = input_dict['gt_bboxes_3d'].tensor
    expected_gt_bboxes_3d = torch.tensor(
        [[9.1724, -1.7559, -1.3550, 0.4800, 1.2000, 1.8900, 0.0505]])
    repr_str = repr(object_noise)
    expected_repr_str = 'ObjectNoise(num_try=100, ' \
                        'translation_std=[0.25, 0.25, 0.25], ' \
                        'global_rot_range=[0.0, 0.0], ' \
                        'rot_range=[-0.15707963267, 0.15707963267])'

    assert repr_str == expected_repr_str
    assert points.shape == (800, 4)
    assert torch.allclose(gt_bboxes_3d, expected_gt_bboxes_3d, 1e-3)
