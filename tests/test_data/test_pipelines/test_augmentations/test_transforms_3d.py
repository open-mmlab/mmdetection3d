# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.bbox import Coord3DMode
from mmdet3d.core.points import DepthPoints, LiDARPoints
from mmdet3d.datasets import (BackgroundPointsFilter, GlobalAlignment,
                              GlobalRotScaleTrans, ObjectNameFilter,
                              ObjectNoise, ObjectRangeFilter, ObjectSample,
                              PointSample, PointShuffle, PointsRangeFilter,
                              RandomDropPointsColor, RandomFlip3D,
                              RandomJitterPoints, VoxelBasedPointSampler)


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
    points = LiDARPoints(points, points_dim=4)
    points = ObjectSample.remove_points_in_boxes(points, boxes)
    assert points.tensor.numpy().shape == (10, 4)


def test_object_sample():
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
    np.random.seed(0)
    object_sample = ObjectSample(db_sampler)
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
    gt_names = annos['name']

    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1).astype(np.float32)
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
    CLASSES = ('Pedestrian', 'Cyclist', 'Car')
    gt_labels = []
    for cat in gt_names:
        if cat in CLASSES:
            gt_labels.append(CLASSES.index(cat))
        else:
            gt_labels.append(-1)
    gt_labels = np.array(gt_labels, dtype=np.long)
    points = LiDARPoints(points, points_dim=4)
    input_dict = dict(
        points=points, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels)
    input_dict = object_sample(input_dict)
    points = input_dict['points']
    gt_bboxes_3d = input_dict['gt_bboxes_3d']
    gt_labels_3d = input_dict['gt_labels_3d']
    repr_str = repr(object_sample)
    expected_repr_str = 'ObjectSample sample_2d=False, ' \
                        'data_root=./tests/data/kitti/, ' \
                        'info_path=./tests/data/kitti/kitti' \
                        '_dbinfos_train.pkl, rate=1.0, ' \
                        'prepare={\'filter_by_difficulty\': [-1], ' \
                        '\'filter_by_min_points\': {\'Pedestrian\': 10}}, ' \
                        'classes=[\'Pedestrian\', \'Cyclist\', \'Car\'], ' \
                        'sample_groups={\'Pedestrian\': 6}'
    assert repr_str == expected_repr_str
    assert points.tensor.numpy().shape == (800, 4)
    assert gt_bboxes_3d.tensor.shape == (1, 7)
    assert np.all(gt_labels_3d == [0])


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
    points = LiDARPoints(points, points_dim=4)
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
    assert points.tensor.numpy().shape == (800, 4)
    assert torch.allclose(gt_bboxes_3d, expected_gt_bboxes_3d, 1e-3)


def test_object_name_filter():
    class_names = ['Pedestrian']
    object_name_filter = ObjectNameFilter(class_names)

    annos = mmcv.load('./tests/data/kitti/kitti_infos_train.pkl')
    info = annos[0]
    rect = info['calib']['R0_rect'].astype(np.float32)
    Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
    annos = info['annos']
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_names = annos['name']

    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1).astype(np.float32)
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
    CLASSES = ('Pedestrian', 'Cyclist', 'Car')
    gt_labels = []
    for cat in gt_names:
        if cat in CLASSES:
            gt_labels.append(CLASSES.index(cat))
        else:
            gt_labels.append(-1)
    gt_labels = np.array(gt_labels, dtype=np.long)
    input_dict = dict(
        gt_bboxes_3d=gt_bboxes_3d.clone(), gt_labels_3d=gt_labels.copy())

    results = object_name_filter(input_dict)
    bboxes_3d = results['gt_bboxes_3d']
    labels_3d = results['gt_labels_3d']
    keep_mask = np.array([name in class_names for name in gt_names])
    assert torch.allclose(gt_bboxes_3d.tensor[keep_mask], bboxes_3d.tensor)
    assert np.all(gt_labels[keep_mask] == labels_3d)

    repr_str = repr(object_name_filter)
    expected_repr_str = f'ObjectNameFilter(classes={class_names})'
    assert repr_str == expected_repr_str


def test_point_shuffle():
    np.random.seed(0)
    torch.manual_seed(0)
    point_shuffle = PointShuffle()

    points = np.fromfile('tests/data/scannet/points/scene0000_00.bin',
                         np.float32).reshape(-1, 6)
    ins_mask = np.fromfile('tests/data/scannet/instance_mask/scene0000_00.bin',
                           np.long)
    sem_mask = np.fromfile('tests/data/scannet/semantic_mask/scene0000_00.bin',
                           np.long)

    points = DepthPoints(
        points.copy(), points_dim=6, attribute_dims=dict(color=[3, 4, 5]))
    input_dict = dict(
        points=points.clone(),
        pts_instance_mask=ins_mask.copy(),
        pts_semantic_mask=sem_mask.copy())
    results = point_shuffle(input_dict)

    shuffle_pts = results['points']
    shuffle_ins_mask = results['pts_instance_mask']
    shuffle_sem_mask = results['pts_semantic_mask']

    shuffle_idx = np.array([
        44, 19, 93, 90, 71, 69, 37, 95, 53, 91, 81, 42, 80, 85, 74, 56, 76, 63,
        82, 40, 26, 92, 57, 10, 16, 66, 89, 41, 97, 8, 31, 24, 35, 30, 65, 7,
        98, 23, 20, 29, 78, 61, 94, 15, 4, 52, 59, 5, 54, 46, 3, 28, 2, 70, 6,
        60, 49, 68, 55, 72, 79, 77, 45, 1, 32, 34, 11, 0, 22, 12, 87, 50, 25,
        47, 36, 96, 9, 83, 62, 84, 18, 17, 75, 67, 13, 48, 39, 21, 64, 88, 38,
        27, 14, 73, 33, 58, 86, 43, 99, 51
    ])
    expected_pts = points.tensor.numpy()[shuffle_idx]
    expected_ins_mask = ins_mask[shuffle_idx]
    expected_sem_mask = sem_mask[shuffle_idx]

    assert np.allclose(shuffle_pts.tensor.numpy(), expected_pts)
    assert np.all(shuffle_ins_mask == expected_ins_mask)
    assert np.all(shuffle_sem_mask == expected_sem_mask)

    repr_str = repr(point_shuffle)
    expected_repr_str = 'PointShuffle'
    assert repr_str == expected_repr_str


def test_points_range_filter():
    pcd_range = [0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
    points_range_filter = PointsRangeFilter(pcd_range)

    points = np.fromfile('tests/data/scannet/points/scene0000_00.bin',
                         np.float32).reshape(-1, 6)
    ins_mask = np.fromfile('tests/data/scannet/instance_mask/scene0000_00.bin',
                           np.long)
    sem_mask = np.fromfile('tests/data/scannet/semantic_mask/scene0000_00.bin',
                           np.long)

    points = DepthPoints(
        points.copy(), points_dim=6, attribute_dims=dict(color=[3, 4, 5]))
    input_dict = dict(
        points=points.clone(),
        pts_instance_mask=ins_mask.copy(),
        pts_semantic_mask=sem_mask.copy())
    results = points_range_filter(input_dict)
    shuffle_pts = results['points']
    shuffle_ins_mask = results['pts_instance_mask']
    shuffle_sem_mask = results['pts_semantic_mask']

    select_idx = np.array(
        [5, 11, 22, 26, 27, 33, 46, 47, 56, 63, 74, 78, 79, 91])
    expected_pts = points.tensor.numpy()[select_idx]
    expected_ins_mask = ins_mask[select_idx]
    expected_sem_mask = sem_mask[select_idx]

    assert np.allclose(shuffle_pts.tensor.numpy(), expected_pts)
    assert np.all(shuffle_ins_mask == expected_ins_mask)
    assert np.all(shuffle_sem_mask == expected_sem_mask)

    repr_str = repr(points_range_filter)
    expected_repr_str = f'PointsRangeFilter(point_cloud_range={pcd_range})'
    assert repr_str == expected_repr_str


def test_object_range_filter():
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    object_range_filter = ObjectRangeFilter(point_cloud_range)

    bbox = np.array(
        [[8.7314, -1.8559, -0.6547, 0.4800, 1.2000, 1.8900, 0.0100],
         [28.7314, -18.559, 0.6547, 2.4800, 1.6000, 1.9200, 5.0100],
         [-2.54, -1.8559, -0.6547, 0.4800, 1.2000, 1.8900, 0.0100],
         [72.7314, -18.559, 0.6547, 6.4800, 11.6000, 4.9200, -0.0100],
         [18.7314, -18.559, 20.6547, 6.4800, 8.6000, 3.9200, -1.0100],
         [3.7314, 42.559, -0.6547, 6.4800, 8.6000, 2.9200, 3.0100]])
    gt_bboxes_3d = LiDARInstance3DBoxes(bbox, origin=(0.5, 0.5, 0.5))
    gt_labels_3d = np.array([0, 2, 1, 1, 2, 0], dtype=np.long)

    input_dict = dict(
        gt_bboxes_3d=gt_bboxes_3d.clone(), gt_labels_3d=gt_labels_3d.copy())
    results = object_range_filter(input_dict)
    bboxes_3d = results['gt_bboxes_3d']
    labels_3d = results['gt_labels_3d']
    keep_mask = np.array([True, True, False, False, True, False])
    expected_bbox = gt_bboxes_3d.tensor[keep_mask]
    expected_bbox[1, 6] -= 2 * np.pi  # limit yaw

    assert torch.allclose(expected_bbox, bboxes_3d.tensor)
    assert np.all(gt_labels_3d[keep_mask] == labels_3d)

    repr_str = repr(object_range_filter)
    expected_repr_str = 'ObjectRangeFilter(point_cloud_range=' \
        '[0.0, -40.0, -3.0, 70.4000015258789, 40.0, 1.0])'
    assert repr_str == expected_repr_str


def test_global_alignment():
    np.random.seed(0)
    global_alignment = GlobalAlignment(rotation_axis=2)

    points = np.fromfile('tests/data/scannet/points/scene0000_00.bin',
                         np.float32).reshape(-1, 6)
    annos = mmcv.load('tests/data/scannet/scannet_infos.pkl')
    info = annos[0]
    axis_align_matrix = info['annos']['axis_align_matrix']

    depth_points = DepthPoints(points.copy(), points_dim=6)

    input_dict = dict(
        points=depth_points.clone(),
        ann_info=dict(axis_align_matrix=axis_align_matrix))

    input_dict = global_alignment(input_dict)
    trans_depth_points = input_dict['points']

    # construct expected transformed points by affine transformation
    pts = np.ones((points.shape[0], 4))
    pts[:, :3] = points[:, :3]
    trans_pts = np.dot(pts, axis_align_matrix.T)
    expected_points = np.concatenate([trans_pts[:, :3], points[:, 3:]], axis=1)

    assert np.allclose(
        trans_depth_points.tensor.numpy(), expected_points, atol=1e-6)

    repr_str = repr(global_alignment)
    expected_repr_str = 'GlobalAlignment(rotation_axis=2)'
    assert repr_str == expected_repr_str


def test_global_rot_scale_trans():
    angle = 0.78539816
    scale = [0.95, 1.05]
    trans_std = 1.0

    # rot_range should be a number or seq of numbers
    with pytest.raises(AssertionError):
        global_rot_scale_trans = GlobalRotScaleTrans(rot_range='0.0')

    # scale_ratio_range should be seq of numbers
    with pytest.raises(AssertionError):
        global_rot_scale_trans = GlobalRotScaleTrans(scale_ratio_range=1.0)

    # translation_std should be a positive number or seq of positive numbers
    with pytest.raises(AssertionError):
        global_rot_scale_trans = GlobalRotScaleTrans(translation_std='0.0')
    with pytest.raises(AssertionError):
        global_rot_scale_trans = GlobalRotScaleTrans(translation_std=-1.0)

    global_rot_scale_trans = GlobalRotScaleTrans(
        rot_range=angle,
        scale_ratio_range=scale,
        translation_std=trans_std,
        shift_height=False)

    np.random.seed(0)
    points = np.fromfile('tests/data/scannet/points/scene0000_00.bin',
                         np.float32).reshape(-1, 6)
    annos = mmcv.load('tests/data/scannet/scannet_infos.pkl')
    info = annos[0]
    gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']

    depth_points = DepthPoints(
        points.copy(), points_dim=6, attribute_dims=dict(color=[3, 4, 5]))
    gt_bboxes_3d = DepthInstance3DBoxes(
        gt_bboxes_3d.copy(),
        box_dim=gt_bboxes_3d.shape[-1],
        with_yaw=False,
        origin=(0.5, 0.5, 0.5))

    input_dict = dict(
        points=depth_points.clone(),
        bbox3d_fields=['gt_bboxes_3d'],
        gt_bboxes_3d=gt_bboxes_3d.clone())

    input_dict = global_rot_scale_trans(input_dict)
    trans_depth_points = input_dict['points']
    trans_bboxes_3d = input_dict['gt_bboxes_3d']

    noise_rot = 0.07667607233534723
    scale_factor = 1.021518936637242
    trans_factor = np.array([0.97873798, 2.2408932, 1.86755799])

    true_depth_points = depth_points.clone()
    true_bboxes_3d = gt_bboxes_3d.clone()
    true_depth_points, noise_rot_mat_T = true_bboxes_3d.rotate(
        noise_rot, true_depth_points)
    true_bboxes_3d.scale(scale_factor)
    true_bboxes_3d.translate(trans_factor)
    true_depth_points.scale(scale_factor)
    true_depth_points.translate(trans_factor)

    assert torch.allclose(
        trans_depth_points.tensor, true_depth_points.tensor, atol=1e-6)
    assert torch.allclose(
        trans_bboxes_3d.tensor, true_bboxes_3d.tensor, atol=1e-6)
    assert input_dict['pcd_scale_factor'] == scale_factor
    assert torch.allclose(
        input_dict['pcd_rotation'], noise_rot_mat_T, atol=1e-6)
    assert np.allclose(input_dict['pcd_trans'], trans_factor)

    repr_str = repr(global_rot_scale_trans)
    expected_repr_str = f'GlobalRotScaleTrans(rot_range={[-angle, angle]},' \
                        f' scale_ratio_range={scale},' \
                        f' translation_std={[trans_std for _ in range(3)]},' \
                        f' shift_height=False)'
    assert repr_str == expected_repr_str

    # points with shift_height but no bbox
    global_rot_scale_trans = GlobalRotScaleTrans(
        rot_range=angle,
        scale_ratio_range=scale,
        translation_std=trans_std,
        shift_height=True)

    # points should have height attribute when shift_height=True
    with pytest.raises(AssertionError):
        input_dict = global_rot_scale_trans(input_dict)

    np.random.seed(0)
    shift_height = points[:, 2:3] * 0.99
    points = np.concatenate([points, shift_height], axis=1)
    depth_points = DepthPoints(
        points.copy(),
        points_dim=7,
        attribute_dims=dict(color=[3, 4, 5], height=6))

    input_dict = dict(points=depth_points.clone(), bbox3d_fields=[])

    input_dict = global_rot_scale_trans(input_dict)
    trans_depth_points = input_dict['points']
    true_shift_height = shift_height * scale_factor

    assert np.allclose(
        trans_depth_points.tensor.numpy(),
        np.concatenate([true_depth_points.tensor.numpy(), true_shift_height],
                       axis=1),
        atol=1e-6)


def test_random_drop_points_color():
    # drop_ratio should be in [0, 1]
    with pytest.raises(AssertionError):
        random_drop_points_color = RandomDropPointsColor(drop_ratio=1.1)

    # 100% drop
    random_drop_points_color = RandomDropPointsColor(drop_ratio=1)

    points = np.fromfile('tests/data/scannet/points/scene0000_00.bin',
                         np.float32).reshape(-1, 6)
    depth_points = DepthPoints(
        points.copy(), points_dim=6, attribute_dims=dict(color=[3, 4, 5]))

    input_dict = dict(points=depth_points.clone())

    input_dict = random_drop_points_color(input_dict)
    trans_depth_points = input_dict['points']
    trans_color = trans_depth_points.color
    assert torch.all(trans_color == trans_color.new_zeros(trans_color.shape))

    # 0% drop
    random_drop_points_color = RandomDropPointsColor(drop_ratio=0)
    input_dict = dict(points=depth_points.clone())

    input_dict = random_drop_points_color(input_dict)
    trans_depth_points = input_dict['points']
    trans_color = trans_depth_points.color
    assert torch.allclose(trans_color, depth_points.tensor[:, 3:6])

    random_drop_points_color = RandomDropPointsColor(drop_ratio=0.5)
    repr_str = repr(random_drop_points_color)
    expected_repr_str = 'RandomDropPointsColor(drop_ratio=0.5)'
    assert repr_str == expected_repr_str


def test_random_flip_3d():
    random_flip_3d = RandomFlip3D(
        flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=1.0)
    points = np.array([[22.7035, 9.3901, -0.2848, 0.0000],
                       [21.9826, 9.1766, -0.2698, 0.0000],
                       [21.4329, 9.0209, -0.2578, 0.0000],
                       [21.3068, 9.0205, -0.2558, 0.0000],
                       [21.3400, 9.1305, -0.2578, 0.0000],
                       [21.3291, 9.2099, -0.2588, 0.0000],
                       [21.2759, 9.2599, -0.2578, 0.0000],
                       [21.2686, 9.2982, -0.2588, 0.0000],
                       [21.2334, 9.3607, -0.2588, 0.0000],
                       [21.2179, 9.4372, -0.2598, 0.0000]])
    bbox3d_fields = ['gt_bboxes_3d']
    img_fields = []
    box_type_3d = LiDARInstance3DBoxes
    gt_bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[38.9229, 18.4417, -1.1459, 0.7100, 1.7600, 1.8600, -2.2652],
             [12.7768, 0.5795, -2.2682, 0.5700, 0.9900, 1.7200, -2.5029],
             [12.7557, 2.2996, -1.4869, 0.6100, 1.1100, 1.9000, -1.9390],
             [10.6677, 0.8064, -1.5435, 0.7900, 0.9600, 1.7900, 1.0856],
             [5.0903, 5.1004, -1.2694, 0.7100, 1.7000, 1.8300, -1.9136]]))
    points = LiDARPoints(points, points_dim=4)
    input_dict = dict(
        points=points,
        bbox3d_fields=bbox3d_fields,
        box_type_3d=box_type_3d,
        img_fields=img_fields,
        gt_bboxes_3d=gt_bboxes_3d)
    input_dict = random_flip_3d(input_dict)
    points = input_dict['points'].tensor.numpy()
    gt_bboxes_3d = input_dict['gt_bboxes_3d'].tensor
    expected_points = np.array([[22.7035, -9.3901, -0.2848, 0.0000],
                                [21.9826, -9.1766, -0.2698, 0.0000],
                                [21.4329, -9.0209, -0.2578, 0.0000],
                                [21.3068, -9.0205, -0.2558, 0.0000],
                                [21.3400, -9.1305, -0.2578, 0.0000],
                                [21.3291, -9.2099, -0.2588, 0.0000],
                                [21.2759, -9.2599, -0.2578, 0.0000],
                                [21.2686, -9.2982, -0.2588, 0.0000],
                                [21.2334, -9.3607, -0.2588, 0.0000],
                                [21.2179, -9.4372, -0.2598, 0.0000]])
    expected_gt_bboxes_3d = torch.tensor(
        [[38.9229, -18.4417, -1.1459, 0.7100, 1.7600, 1.8600, 5.4068],
         [12.7768, -0.5795, -2.2682, 0.5700, 0.9900, 1.7200, 5.6445],
         [12.7557, -2.2996, -1.4869, 0.6100, 1.1100, 1.9000, 5.0806],
         [10.6677, -0.8064, -1.5435, 0.7900, 0.9600, 1.7900, 2.0560],
         [5.0903, -5.1004, -1.2694, 0.7100, 1.7000, 1.8300, 5.0552]])
    repr_str = repr(random_flip_3d)
    expected_repr_str = 'RandomFlip3D(sync_2d=True,' \
                        ' flip_ratio_bev_vertical=1.0)'
    assert np.allclose(points, expected_points)
    assert torch.allclose(gt_bboxes_3d, expected_gt_bboxes_3d)
    assert repr_str == expected_repr_str


def test_random_jitter_points():
    # jitter_std should be a number or seq of numbers
    with pytest.raises(AssertionError):
        random_jitter_points = RandomJitterPoints(jitter_std='0.0')

    # clip_range should be a number or seq of numbers
    with pytest.raises(AssertionError):
        random_jitter_points = RandomJitterPoints(clip_range='0.0')

    random_jitter_points = RandomJitterPoints(jitter_std=0.01, clip_range=0.05)
    np.random.seed(0)
    points = np.fromfile('tests/data/scannet/points/scene0000_00.bin',
                         np.float32).reshape(-1, 6)[:10]
    depth_points = DepthPoints(
        points.copy(), points_dim=6, attribute_dims=dict(color=[3, 4, 5]))

    input_dict = dict(points=depth_points.clone())

    input_dict = random_jitter_points(input_dict)
    trans_depth_points = input_dict['points']

    jitter_noise = np.array([[0.01764052, 0.00400157, 0.00978738],
                             [0.02240893, 0.01867558, -0.00977278],
                             [0.00950088, -0.00151357, -0.00103219],
                             [0.00410598, 0.00144044, 0.01454273],
                             [0.00761038, 0.00121675, 0.00443863],
                             [0.00333674, 0.01494079, -0.00205158],
                             [0.00313068, -0.00854096, -0.0255299],
                             [0.00653619, 0.00864436, -0.00742165],
                             [0.02269755, -0.01454366, 0.00045759],
                             [-0.00187184, 0.01532779, 0.01469359]])

    trans_depth_points = trans_depth_points.tensor.numpy()
    expected_depth_points = points
    expected_depth_points[:, :3] += jitter_noise
    assert np.allclose(trans_depth_points, expected_depth_points)

    repr_str = repr(random_jitter_points)
    jitter_std = [0.01, 0.01, 0.01]
    clip_range = [-0.05, 0.05]
    expected_repr_str = f'RandomJitterPoints(jitter_std={jitter_std},' \
                        f' clip_range={clip_range})'
    assert repr_str == expected_repr_str

    # test clipping very large noise
    random_jitter_points = RandomJitterPoints(jitter_std=1.0, clip_range=0.05)
    input_dict = dict(points=depth_points.clone())

    input_dict = random_jitter_points(input_dict)
    trans_depth_points = input_dict['points']
    assert (trans_depth_points.tensor - depth_points.tensor).max().item() <= \
        0.05 + 1e-6
    assert (trans_depth_points.tensor - depth_points.tensor).min().item() >= \
        -0.05 - 1e-6


def test_background_points_filter():
    np.random.seed(0)
    background_points_filter = BackgroundPointsFilter((0.5, 2.0, 0.5))
    points = np.fromfile(
        './tests/data/kitti/training/velodyne_reduced/000000.bin',
        np.float32).reshape(-1, 4)
    orig_points = points.copy()
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
    extra_points = gt_bboxes_3d.corners.reshape(8, 3)[[1, 2, 5, 6], :]
    extra_points[:, 2] += 0.1
    extra_points = torch.cat([extra_points, extra_points.new_zeros(4, 1)], 1)
    points = np.concatenate([points, extra_points.numpy()], 0)
    points = LiDARPoints(points, points_dim=4)
    input_dict = dict(points=points, gt_bboxes_3d=gt_bboxes_3d)
    origin_gt_bboxes_3d = gt_bboxes_3d.clone()
    input_dict = background_points_filter(input_dict)

    points = input_dict['points'].tensor.numpy()
    repr_str = repr(background_points_filter)
    expected_repr_str = 'BackgroundPointsFilter(bbox_enlarge_range=' \
                        '[[0.5, 2.0, 0.5]])'
    assert repr_str == expected_repr_str
    assert points.shape == (800, 4)
    assert np.equal(orig_points, points).all()
    assert np.equal(input_dict['gt_bboxes_3d'].tensor.numpy(),
                    origin_gt_bboxes_3d.tensor.numpy()).all()

    # test single float config
    BackgroundPointsFilter(0.5)

    # The length of bbox_enlarge_range should be 3
    with pytest.raises(AssertionError):
        BackgroundPointsFilter((0.5, 2.0))


def test_voxel_based_point_filter():
    np.random.seed(0)
    cur_sweep_cfg = dict(
        voxel_size=[0.1, 0.1, 0.1],
        point_cloud_range=[-50, -50, -4, 50, 50, 2],
        max_num_points=1,
        max_voxels=1024)
    prev_sweep_cfg = dict(
        voxel_size=[0.1, 0.1, 0.1],
        point_cloud_range=[-50, -50, -4, 50, 50, 2],
        max_num_points=1,
        max_voxels=1024)
    voxel_based_points_filter = VoxelBasedPointSampler(
        cur_sweep_cfg, prev_sweep_cfg, time_dim=3)
    points = np.stack([
        np.random.rand(4096) * 120 - 60,
        np.random.rand(4096) * 120 - 60,
        np.random.rand(4096) * 10 - 6
    ],
                      axis=-1)

    input_time = np.concatenate([np.zeros([2048, 1]), np.ones([2048, 1])], 0)
    input_points = np.concatenate([points, input_time], 1)
    input_points = LiDARPoints(input_points, points_dim=4)
    input_dict = dict(
        points=input_points, pts_mask_fields=[], pts_seg_fields=[])
    input_dict = voxel_based_points_filter(input_dict)

    points = input_dict['points']
    repr_str = repr(voxel_based_points_filter)
    expected_repr_str = """VoxelBasedPointSampler(
    num_cur_sweep=1024,
    num_prev_sweep=1024,
    time_dim=3,
    cur_voxel_generator=
        VoxelGenerator(voxel_size=[0.1 0.1 0.1],
                       point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 2.0],
                       max_num_points=1,
                       max_voxels=1024,
                       grid_size=[1000, 1000, 60]),
    prev_voxel_generator=
        VoxelGenerator(voxel_size=[0.1 0.1 0.1],
                       point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 2.0],
                       max_num_points=1,
                       max_voxels=1024,
                       grid_size=[1000, 1000, 60]))"""

    assert repr_str == expected_repr_str
    assert points.shape == (2048, 4)
    assert (points.tensor[:, :3].min(0)[0].numpy() <
            cur_sweep_cfg['point_cloud_range'][0:3]).sum() == 0
    assert (points.tensor[:, :3].max(0)[0].numpy() >
            cur_sweep_cfg['point_cloud_range'][3:6]).sum() == 0

    # Test instance mask and semantic mask
    input_dict = dict(points=input_points)
    input_dict['pts_instance_mask'] = np.random.randint(0, 10, [4096])
    input_dict['pts_semantic_mask'] = np.random.randint(0, 6, [4096])
    input_dict['pts_mask_fields'] = ['pts_instance_mask']
    input_dict['pts_seg_fields'] = ['pts_semantic_mask']

    input_dict = voxel_based_points_filter(input_dict)
    pts_instance_mask = input_dict['pts_instance_mask']
    pts_semantic_mask = input_dict['pts_semantic_mask']
    assert pts_instance_mask.shape == (2048, )
    assert pts_semantic_mask.shape == (2048, )
    assert pts_instance_mask.max() < 10
    assert pts_instance_mask.min() >= 0
    assert pts_semantic_mask.max() < 6
    assert pts_semantic_mask.min() >= 0


def test_points_sample():
    np.random.seed(0)
    points = np.fromfile(
        './tests/data/kitti/training/velodyne_reduced/000000.bin',
        np.float32).reshape(-1, 4)
    annos = mmcv.load('./tests/data/kitti/kitti_infos_train.pkl')
    info = annos[0]
    rect = torch.tensor(info['calib']['R0_rect'].astype(np.float32))
    Trv2c = torch.tensor(info['calib']['Tr_velo_to_cam'].astype(np.float32))

    points = LiDARPoints(
        points.copy(), points_dim=4).convert_to(Coord3DMode.CAM, rect @ Trv2c)
    num_points = 20
    sample_range = 40
    input_dict = dict(points=points.clone())

    point_sample = PointSample(
        num_points=num_points, sample_range=sample_range)
    sampled_pts = point_sample(input_dict)['points']

    select_idx = np.array([
        622, 146, 231, 444, 504, 533, 80, 401, 379, 2, 707, 562, 176, 491, 496,
        464, 15, 590, 194, 449
    ])
    expected_pts = points.tensor.numpy()[select_idx]
    assert np.allclose(sampled_pts.tensor.numpy(), expected_pts)

    repr_str = repr(point_sample)
    expected_repr_str = f'PointSample(num_points={num_points}, ' \
                        f'sample_range={sample_range}, ' \
                        'replace=False)'
    assert repr_str == expected_repr_str

    # test when number of far points are larger than number of sampled points
    np.random.seed(0)
    point_sample = PointSample(num_points=2, sample_range=sample_range)
    input_dict = dict(points=points.clone())
    sampled_pts = point_sample(input_dict)['points']

    select_idx = np.array([449, 444])
    expected_pts = points.tensor.numpy()[select_idx]
    assert np.allclose(sampled_pts.tensor.numpy(), expected_pts)
