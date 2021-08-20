# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import tempfile
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


def test_show():
    import mmcv
    from os import path as osp

    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    eval_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ]
    nus_dataset = NuScenesDataset('tests/data/nuscenes/nus_info.pkl', None,
                                  'tests/data/nuscenes')
    boxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[46.1218, -4.6496, -0.9275, 0.5316, 1.4442, 1.7450, 1.1749],
             [33.3189, 0.1981, 0.3136, 0.5656, 1.2301, 1.7985, 1.5723],
             [46.1366, -4.6404, -0.9510, 0.5162, 1.6501, 1.7540, 1.3778],
             [33.2646, 0.2297, 0.3446, 0.5746, 1.3365, 1.7947, 1.5430],
             [58.9079, 16.6272, -1.5829, 1.5656, 3.9313, 1.4899, 1.5505]]))
    scores_3d = torch.tensor([0.1815, 0.1663, 0.5792, 0.2194, 0.2780])
    labels_3d = torch.tensor([0, 0, 1, 1, 2])
    result = dict(boxes_3d=boxes_3d, scores_3d=scores_3d, labels_3d=labels_3d)
    results = [dict(pts_bbox=result)]
    nus_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    file_name = 'n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470948018'
    pts_file_path = osp.join(temp_dir, file_name, f'{file_name}_points.obj')
    gt_file_path = osp.join(temp_dir, file_name, f'{file_name}_gt.obj')
    pred_file_path = osp.join(temp_dir, file_name, f'{file_name}_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()
