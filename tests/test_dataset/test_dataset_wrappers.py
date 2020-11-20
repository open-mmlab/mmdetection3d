import numpy as np
import torch

from mmdet3d.datasets.builder import build_dataset


def test_getitem():
    np.random.seed(1)
    torch.manual_seed(1)
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
            sweeps_num=9,
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=file_client_args,
            pad_empty_sweeps=True,
            remove_close=True,
            test_mode=True),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        # dict(type='ObjectSample', db_sampler=db_sampler),
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
    input_modality = dict(
        use_lidar=True,
        use_camera=False,
        use_radar=False,
        use_map=False,
        use_external=False)
    dataset_cfg = dict(
        type='CBGSDataset',
        dataset=dict(
            type='NuScenesDataset',
            data_root='tests/data/nuscenes',
            ann_file='tests/data/nuscenes/nus_info.pkl',
            pipeline=pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'))
    nus_dataset = build_dataset(dataset_cfg)
    assert len(nus_dataset) == 20

    data = nus_dataset[0]
    assert data['img_metas'].data['flip'] is True
    assert data['img_metas'].data['pcd_horizontal_flip'] is True
    assert data['points']._data.shape == (537, 5)

    data = nus_dataset[2]
    assert data['img_metas'].data['flip'] is False
    assert data['img_metas'].data['pcd_horizontal_flip'] is False
    assert data['points']._data.shape == (901, 5)
