import numpy as np
import torch

from mmdet3d.datasets import NuScenesDataset


def test_getitem():
    np.random.seed(0)
    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    file_client_args = dict(backend='disk')
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            load_dim=5,
            use_dim=5,
            file_client_args=file_client_args),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=file_client_args),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectNameFilter', classes=class_names),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

    nus_dataset = NuScenesDataset('tests/data/nuscenes/nus_info.pkl', pipeline,
                                  'tests/data/nuscenes')
    data = nus_dataset[0]
    assert data['img_metas'].data['flip'] is True
    assert data['img_metas'].data['pcd_horizontal_flip'] is True
    assert data['points']._data.shape == (100, 4)
    assert data['gt_bboxes_3d']._data.tensor.shape == (35, 9)
    assert data['gt_labels_3d']._data.shape == torch.Size([35])


# def test_evaluate():
#     nus_dataset = NuScenesDataset('tests/data/nuscenes/nus_info.pkl',
#     None, 'tests/data/nuscenes')
#     boxes_3d = LiDARInstance3DBoxes(
#         torch.tensor([[-8.1570e+00,  8.7116e+00, -1.8850e+00,  1.7631e+00,
#         4.3486e+00,
#           1.5210e+00, -1.2581e+00,  6.9662e-01, -2.1322e-01],
#         [-1.7330e+01,  7.9215e+00, -1.5796e+00,  1.7468e+00,  4.3394e+00,
#           1.5599e+00,  1.5385e+00,  0.0000e+00, -0.0000e+00],
#         [-2.1571e+00,  5.7065e+00, -2.0842e+00,  1.8091e+00,  4.3956e+00,
#           1.4373e+00, -5.4396e-01, -3.1101e-02,  5.3448e-02],
#         [-1.1666e+01,  3.0981e+01, -2.3282e+00,  1.7693e+00,  4.5989e+00,
#           1.8377e+00,  1.5523e+00, -1.6222e-01, -2.8880e-03],
#         [-2.2412e+01, -4.1056e+01,  4.3405e-02,  7.0996e-01,  1.1186e+00,
#           1.8183e+00, -1.9027e+00,  1.6083e+00,  6.4692e-01]]), 9)
#     labels_3d = torch.tensor([0, 0, 0, 0, 7])
#     scores_3d = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
#     result = dict(
#         pts_bbox=dict(
#         boxes_3d = boxes_3d,
#         labels_3d = labels_3d,
#         scores_3d = scores_3d
#     ))
#     nus_dataset.evaluate([result])
