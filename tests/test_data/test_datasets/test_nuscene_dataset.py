import numpy as np

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
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=file_client_args),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=2,
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

    nus_dataset = NuScenesDataset(
        'tests/data/nuscenes/nus_info.pkl',
        pipeline,
        'tests/data/nuscenes',
        test_mode=True)
    data = nus_dataset[0]
    assert data['img_metas'][0].data['flip'] is False
    assert data['img_metas'][0].data['pcd_horizontal_flip'] is False
    assert data['points'][0]._data.shape == (100, 4)

    data = nus_dataset[1]
    assert data['img_metas'][0].data['flip'] is False
    assert data['img_metas'][0].data['pcd_horizontal_flip'] is False
    assert data['points'][0]._data.shape == (597, 4)
