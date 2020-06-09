import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose


def test_outdoor_pipeline():
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    class_names = ['Car']
    np.random.seed(0)

    train_pipeline = [
        dict(type='LoadPointsFromFile', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='ObjectNoise',
            num_try=100,
            loc_noise_std=[1.0, 1.0, 0.5],
            global_rot_range=[0.0, 0.0],
            rot_uniform_noise=[-0.78539816, 0.78539816]),
        dict(type='RandomFlip3D', flip_ratio=0.5),
        dict(
            type='GlobalRotScale',
            rot_uniform_noise=[-0.78539816, 0.78539816],
            scaling_uniform_noise=[0.95, 1.05]),
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
    )

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
