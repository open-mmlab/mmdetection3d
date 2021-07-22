import mmcv
import numpy as np
import torch
from os import path as osp

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose


def test_scannet_pipeline():
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk',
                   'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                   'sink', 'bathtub', 'garbagebin')

    np.random.seed(0)
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_mask_3d=True,
            with_seg_3d=True),
        dict(type='GlobalAlignment', rotation_axis=2),
        dict(
            type='PointSegClassMapping',
            valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33,
                           34, 36, 39)),
        dict(type='IndoorPointSample', num_points=5),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=1.0,
            flip_ratio_bev_vertical=1.0),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.087266, 0.087266],
            scale_ratio_range=[1.0, 1.0],
            shift_height=True),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
                'pts_instance_mask'
            ]),
    ]
    pipeline = Compose(pipelines)
    info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')[0]
    results = dict()
    data_path = './tests/data/scannet'
    results['pts_filename'] = osp.join(data_path, info['pts_path'])
    if info['annos']['gt_num'] != 0:
        scannet_gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
            np.float32)
        scannet_gt_labels_3d = info['annos']['class'].astype(np.long)
    else:
        scannet_gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
        scannet_gt_labels_3d = np.zeros((1, ), dtype=np.long)
    results['ann_info'] = dict()
    results['ann_info']['pts_instance_mask_path'] = osp.join(
        data_path, info['pts_instance_mask_path'])
    results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, info['pts_semantic_mask_path'])
    results['ann_info']['gt_bboxes_3d'] = DepthInstance3DBoxes(
        scannet_gt_bboxes_3d, box_dim=6, with_yaw=False)
    results['ann_info']['gt_labels_3d'] = scannet_gt_labels_3d
    results['ann_info']['axis_align_matrix'] = \
        info['annos']['axis_align_matrix']

    results['img_fields'] = []
    results['bbox3d_fields'] = []
    results['pts_mask_fields'] = []
    results['pts_seg_fields'] = []

    results = pipeline(results)

    points = results['points']._data
    gt_bboxes_3d = results['gt_bboxes_3d']._data
    gt_labels_3d = results['gt_labels_3d']._data
    pts_semantic_mask = results['pts_semantic_mask']._data
    pts_instance_mask = results['pts_instance_mask']._data
    expected_points = torch.tensor(
        [[1.8339e+00, 2.1093e+00, 2.2900e+00, 2.3895e+00],
         [3.6079e+00, 1.4592e-01, 2.0687e+00, 2.1682e+00],
         [4.1886e+00, 5.0614e+00, -1.0841e-01, -8.8736e-03],
         [6.8790e+00, 1.5086e+00, -9.3154e-02, 6.3816e-03],
         [4.8253e+00, 2.6668e-01, 1.4917e+00, 1.5912e+00]])
    expected_gt_bboxes_3d = torch.tensor(
        [[-1.1835, -3.6317, 1.8565, 1.7577, 0.3761, 0.5724, 0.0000],
         [-3.1832, 3.2269, 1.5268, 0.6727, 0.2251, 0.6715, 0.0000],
         [-0.9598, -2.2864, 0.6165, 0.7506, 2.5709, 1.2145, 0.0000],
         [-2.6988, -2.7354, 0.9722, 0.7680, 1.8877, 0.2870, 0.0000],
         [3.2989, 0.2885, 1.0712, 0.7600, 3.8814, 2.1603, 0.0000]])
    expected_gt_labels_3d = np.array([
        6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0,
        0, 0, 0, 5, 5, 5
    ])
    expected_pts_semantic_mask = np.array([0, 18, 18, 18, 18])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    assert torch.allclose(points, expected_points, 1e-2)
    assert torch.allclose(gt_bboxes_3d.tensor[:5, :], expected_gt_bboxes_3d,
                          1e-2)
    assert np.all(gt_labels_3d.numpy() == expected_gt_labels_3d)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert np.all(pts_instance_mask.numpy() == expected_pts_instance_mask)


def test_scannet_seg_pipeline():
    class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                   'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                   'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                   'sink', 'bathtub', 'otherfurniture')

    np.random.seed(0)
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            use_color=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True),
        dict(
            type='PointSegClassMapping',
            valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24,
                           28, 33, 34, 36, 39),
            max_cat_id=40),
        dict(
            type='IndoorPatchPointSample',
            num_points=5,
            block_size=1.5,
            ignore_index=len(class_names),
            use_normalized_coord=True,
            enlarge_size=0.2,
            min_unique_num=None),
        dict(type='NormalizePointsColor', color_mean=None),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
    ]
    pipeline = Compose(pipelines)
    info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')[0]
    results = dict()
    data_path = './tests/data/scannet'
    results['pts_filename'] = osp.join(data_path, info['pts_path'])
    results['ann_info'] = dict()
    results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, info['pts_semantic_mask_path'])

    results['pts_seg_fields'] = []

    results = pipeline(results)

    points = results['points']._data
    pts_semantic_mask = results['pts_semantic_mask']._data

    # build sampled points
    scannet_points = np.fromfile(
        osp.join(data_path, info['pts_path']), dtype=np.float32).reshape(
            (-1, 6))
    scannet_choices = np.array([87, 34, 58, 9, 18])
    scannet_center = np.array([-2.1772466, -3.4789145, 1.242711])
    scannet_center[2] = 0.0
    scannet_coord_max = np.amax(scannet_points[:, :3], axis=0)
    expected_points = np.concatenate([
        scannet_points[scannet_choices, :3] - scannet_center,
        scannet_points[scannet_choices, 3:] / 255.,
        scannet_points[scannet_choices, :3] / scannet_coord_max
    ],
                                     axis=1)
    expected_pts_semantic_mask = np.array([13, 13, 12, 2, 0])
    assert np.allclose(points.numpy(), expected_points, atol=1e-6)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)


def test_s3dis_seg_pipeline():
    class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                   'door', 'table', 'chair', 'sofa', 'bookcase', 'board',
                   'clutter')

    np.random.seed(0)
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            use_color=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True),
        dict(
            type='PointSegClassMapping',
            valid_cat_ids=tuple(range(len(class_names))),
            max_cat_id=13),
        dict(
            type='IndoorPatchPointSample',
            num_points=5,
            block_size=1.0,
            ignore_index=len(class_names),
            use_normalized_coord=True,
            enlarge_size=0.2,
            min_unique_num=None),
        dict(type='NormalizePointsColor', color_mean=None),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
    ]
    pipeline = Compose(pipelines)
    info = mmcv.load('./tests/data/s3dis/s3dis_infos.pkl')[0]
    results = dict()
    data_path = './tests/data/s3dis'
    results['pts_filename'] = osp.join(data_path, info['pts_path'])
    results['ann_info'] = dict()
    results['ann_info']['pts_semantic_mask_path'] = osp.join(
        data_path, info['pts_semantic_mask_path'])

    results['pts_seg_fields'] = []

    results = pipeline(results)

    points = results['points']._data
    pts_semantic_mask = results['pts_semantic_mask']._data

    # build sampled points
    s3dis_points = np.fromfile(
        osp.join(data_path, info['pts_path']), dtype=np.float32).reshape(
            (-1, 6))
    s3dis_choices = np.array([87, 37, 60, 18, 31])
    s3dis_center = np.array([2.691, 2.231, 3.172])
    s3dis_center[2] = 0.0
    s3dis_coord_max = np.amax(s3dis_points[:, :3], axis=0)
    expected_points = np.concatenate([
        s3dis_points[s3dis_choices, :3] - s3dis_center,
        s3dis_points[s3dis_choices, 3:] / 255.,
        s3dis_points[s3dis_choices, :3] / s3dis_coord_max
    ],
                                     axis=1)
    expected_pts_semantic_mask = np.array([0, 1, 0, 8, 0])
    assert np.allclose(points.numpy(), expected_points, atol=1e-6)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)


def test_sunrgbd_pipeline():
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                   'dresser', 'night_stand', 'bookshelf', 'bathtub')
    np.random.seed(0)
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(type='LoadAnnotations3D'),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=1.0,
        ),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.523599, 0.523599],
            scale_ratio_range=[0.85, 1.15],
            shift_height=True),
        dict(type='IndoorPointSample', num_points=5),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
    ]
    pipeline = Compose(pipelines)
    results = dict()
    info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')[0]
    data_path = './tests/data/sunrgbd'
    results['pts_filename'] = osp.join(data_path, info['pts_path'])

    if info['annos']['gt_num'] != 0:
        gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
            np.float32)
        gt_labels_3d = info['annos']['class'].astype(np.long)
    else:
        gt_bboxes_3d = np.zeros((1, 7), dtype=np.float32)
        gt_labels_3d = np.zeros((1, ), dtype=np.long)

    # prepare input of pipeline
    results['ann_info'] = dict()
    results['ann_info']['gt_bboxes_3d'] = DepthInstance3DBoxes(gt_bboxes_3d)
    results['ann_info']['gt_labels_3d'] = gt_labels_3d
    results['img_fields'] = []
    results['bbox3d_fields'] = []
    results['pts_mask_fields'] = []
    results['pts_seg_fields'] = []

    results = pipeline(results)
    points = results['points']._data
    gt_bboxes_3d = results['gt_bboxes_3d']._data
    gt_labels_3d = results['gt_labels_3d']._data
    expected_points = torch.tensor([[0.8678, 1.3470, 0.1105, 0.0905],
                                    [0.8707, 1.3635, 0.0437, 0.0238],
                                    [0.8636, 1.3511, 0.0504, 0.0304],
                                    [0.8690, 1.3461, 0.1265, 0.1065],
                                    [0.8668, 1.3434, 0.1216, 0.1017]])
    expected_gt_bboxes_3d = torch.tensor(
        [[-1.2136, 4.0206, -0.2412, 2.2493, 1.8444, 1.9245, 1.3989],
         [-2.7420, 4.5777, -0.7686, 0.5718, 0.8629, 0.9510, 1.4446],
         [0.9729, 1.9087, -0.1443, 0.6965, 1.5273, 2.0563, 2.9924]])
    expected_gt_labels_3d = np.array([0, 7, 6])
    assert torch.allclose(gt_bboxes_3d.tensor, expected_gt_bboxes_3d, 1e-3)
    assert np.allclose(gt_labels_3d.flatten(), expected_gt_labels_3d)
    assert torch.allclose(points, expected_points, 1e-2)
