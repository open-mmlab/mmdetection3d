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
