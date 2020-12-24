import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose


def test_outdoor_aug_pipeline():
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    class_names = ['Car']
    np.random.seed(0)

    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='ObjectNoise',
            num_try=100,
            translation_std=[1.0, 1.0, 0.5],
            global_rot_range=[0.0, 0.0],
            rot_range=[-0.78539816, 0.78539816]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    pipeline = Compose(train_pipeline)

    gt_bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor([
            [
                2.16902428e+01, -4.06038128e-02, -1.61906636e+00,
                1.65999997e+00, 3.20000005e+00, 1.61000001e+00, -1.53999996e+00
            ],
            [
                7.05006886e+00, -6.57459593e+00, -1.60107934e+00,
                2.27999997e+00, 1.27799997e+01, 3.66000009e+00, 1.54999995e+00
            ],
            [
                2.24698811e+01, -6.69203758e+00, -1.50118136e+00,
                2.31999993e+00, 1.47299995e+01, 3.64000010e+00, 1.59000003e+00
            ],
            [
                3.48291969e+01, -7.09058380e+00, -1.36622977e+00,
                2.31999993e+00, 1.00400000e+01, 3.60999990e+00, 1.61000001e+00
            ],
            [
                4.62394600e+01, -7.75838804e+00, -1.32405007e+00,
                2.33999991e+00, 1.28299999e+01, 3.63000011e+00, 1.63999999e+00
            ],
            [
                2.82966995e+01, -5.55755794e-01, -1.30332506e+00,
                1.47000003e+00, 2.23000002e+00, 1.48000002e+00, -1.57000005e+00
            ],
            [
                2.66690197e+01, 2.18230209e+01, -1.73605704e+00,
                1.55999994e+00, 3.48000002e+00, 1.39999998e+00, -1.69000006e+00
            ],
            [
                3.13197803e+01, 8.16214371e+00, -1.62177873e+00,
                1.74000001e+00, 3.76999998e+00, 1.48000002e+00, 2.78999996e+00
            ],
            [
                4.34395561e+01, -1.95209332e+01, -1.20757008e+00,
                1.69000006e+00, 4.09999990e+00, 1.40999997e+00, -1.53999996e+00
            ],
            [
                3.29882965e+01, -3.79360509e+00, -1.69245458e+00,
                1.74000001e+00, 4.09000015e+00, 1.49000001e+00, -1.52999997e+00
            ],
            [
                3.85469360e+01, 8.35060215e+00, -1.31423414e+00,
                1.59000003e+00, 4.28000021e+00, 1.45000005e+00, 1.73000002e+00
            ],
            [
                2.22492104e+01, -1.13536005e+01, -1.38272512e+00,
                1.62000000e+00, 3.55999994e+00, 1.71000004e+00, 2.48000002e+00
            ],
            [
                3.36115799e+01, -1.97708054e+01, -4.92827654e-01,
                1.64999998e+00, 3.54999995e+00, 1.79999995e+00, -1.57000005e+00
            ],
            [
                9.85029602e+00, -1.51294518e+00, -1.66834795e+00,
                1.59000003e+00, 3.17000008e+00, 1.38999999e+00, -8.39999974e-01
            ]
        ],
                     dtype=torch.float32))
    gt_labels_3d = np.array([0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    results = dict(
        pts_filename='tests/data/kitti/a.bin',
        ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d),
        bbox3d_fields=[],
        img_fields=[])

    output = pipeline(results)

    expected_tensor = torch.tensor(
        [[20.6514, -8.8250, -1.0816, 1.5893, 3.0637, 1.5414, -1.9216],
         [7.9374, 4.9457, -1.2008, 2.1829, 12.2357, 3.5041, 1.6629],
         [20.8115, -2.0273, -1.8893, 2.2212, 14.1026, 3.4850, 2.6513],
         [32.3850, -5.2135, -1.1321, 2.2212, 9.6124, 3.4562, 2.6498],
         [43.7022, -7.8316, -0.5090, 2.2403, 12.2836, 3.4754, 2.0146],
         [25.3300, -9.6670, -1.0855, 1.4074, 2.1350, 1.4170, -0.7141],
         [16.5414, -29.0583, -0.9768, 1.4936, 3.3318, 1.3404, -0.7153],
         [24.6548, -18.9226, -1.3567, 1.6659, 3.6094, 1.4170, 1.3970],
         [45.8403, 1.8183, -1.1626, 1.6180, 3.9254, 1.3499, -0.6886],
         [30.6288, -8.4497, -1.4881, 1.6659, 3.9158, 1.4265, -0.7241],
         [32.3316, -22.4611, -1.3131, 1.5223, 4.0977, 1.3882, 2.4186],
         [22.4492, 3.2944, -2.1674, 1.5510, 3.4084, 1.6372, 0.3928],
         [37.3824, 5.0472, -0.6579, 1.5797, 3.3988, 1.7233, -1.4862],
         [8.9259, -1.2578, -1.6081, 1.5223, 3.0350, 1.3308, -1.7212]])
    assert torch.allclose(
        output['gt_bboxes_3d']._data.tensor, expected_tensor, atol=1e-3)


def test_outdoor_velocity_aug_pipeline():
    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    np.random.seed(0)

    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    pipeline = Compose(train_pipeline)

    gt_bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[
                -5.2422e+00, 4.0021e+01, -4.7643e-01, 2.0620e+00, 4.4090e+00,
                1.5480e+00, -1.4880e+00, 8.5338e-03, 4.4934e-02
            ],
             [
                 -2.6675e+01, 5.5950e+00, -1.3053e+00, 3.4300e-01, 4.5800e-01,
                 7.8200e-01, -4.6276e+00, -4.3284e-04, -1.8543e-03
             ],
             [
                 -5.8098e+00, 3.5409e+01, -6.6511e-01, 2.3960e+00, 3.9690e+00,
                 1.7320e+00, -4.6520e+00, 0.0000e+00, 0.0000e+00
             ],
             [
                 -3.1309e+01, 1.0901e+00, -1.0561e+00, 1.9440e+00, 3.8570e+00,
                 1.7230e+00, -2.8143e+00, -2.7606e-02, -8.0573e-02
             ],
             [
                 -4.5642e+01, 2.0136e+01, -2.4681e-02, 1.9870e+00, 4.4400e+00,
                 1.9420e+00, 2.8336e-01, 0.0000e+00, 0.0000e+00
             ],
             [
                 -5.1617e+00, 1.8305e+01, -1.0879e+00, 2.3230e+00, 4.8510e+00,
                 1.3710e+00, -1.5803e+00, 0.0000e+00, 0.0000e+00
             ],
             [
                 -2.5285e+01, 4.1442e+00, -1.2713e+00, 1.7550e+00, 1.9890e+00,
                 2.2200e+00, -4.4900e+00, -3.1784e-02, -1.5291e-01
             ],
             [
                 -2.2611e+00, 1.9170e+01, -1.1452e+00, 9.1900e-01, 1.1230e+00,
                 1.9310e+00, 4.7790e-02, 6.7684e-02, -1.7537e+00
             ],
             [
                 -6.5878e+01, 1.3500e+01, -2.2528e-01, 1.8200e+00, 3.8520e+00,
                 1.5450e+00, -2.8757e+00, 0.0000e+00, 0.0000e+00
             ],
             [
                 -5.4490e+00, 2.8363e+01, -7.7275e-01, 2.2360e+00, 3.7540e+00,
                 1.5590e+00, -4.6520e+00, -7.9736e-03, 7.7207e-03
             ]],
            dtype=torch.float32),
        box_dim=9)

    gt_labels_3d = np.array([0, 8, 0, 0, 0, 0, -1, 7, 0, 0])
    results = dict(
        pts_filename='tests/data/kitti/a.bin',
        ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d),
        bbox3d_fields=[],
        img_fields=[])

    output = pipeline(results)

    expected_tensor = torch.tensor(
        [[
            -3.7849e+00, -4.1057e+01, -4.8668e-01, 2.1064e+00, 4.5039e+00,
            1.5813e+00, -1.6919e+00, 1.0469e-02, -4.5533e-02
        ],
         [
             -2.7010e+01, -6.7551e+00, -1.3334e+00, 3.5038e-01, 4.6786e-01,
             7.9883e-01, 1.4477e+00, -5.1440e-04, 1.8758e-03
         ],
         [
             -4.5448e+00, -3.6372e+01, -6.7942e-01, 2.4476e+00, 4.0544e+00,
             1.7693e+00, 1.4721e+00, 0.0000e+00, -0.0000e+00
         ],
         [
             -3.1916e+01, -2.3379e+00, -1.0788e+00, 1.9858e+00, 3.9400e+00,
             1.7601e+00, -3.6564e-01, -3.1333e-02, 8.1166e-02
         ],
         [
             -4.5802e+01, -2.2340e+01, -2.5213e-02, 2.0298e+00, 4.5355e+00,
             1.9838e+00, 2.8199e+00, 0.0000e+00, -0.0000e+00
         ],
         [
             -4.5526e+00, -1.8887e+01, -1.1114e+00, 2.3730e+00, 4.9554e+00,
             1.4005e+00, -1.5997e+00, 0.0000e+00, -0.0000e+00
         ],
         [
             -2.5648e+01, -5.2197e+00, -1.2987e+00, 1.7928e+00, 2.0318e+00,
             2.2678e+00, 1.3100e+00, -3.8428e-02, 1.5485e-01
         ],
         [
             -1.5578e+00, -1.9657e+01, -1.1699e+00, 9.3878e-01, 1.1472e+00,
             1.9726e+00, 3.0555e+00, 4.5907e-04, 1.7928e+00
         ],
         [
             -4.4522e+00, -2.9166e+01, -7.8938e-01, 2.2841e+00, 3.8348e+00,
             1.5925e+00, 1.4721e+00, -7.8371e-03, -8.1931e-03
         ]])
    assert torch.allclose(
        output['gt_bboxes_3d']._data.tensor, expected_tensor, atol=1e-3)
