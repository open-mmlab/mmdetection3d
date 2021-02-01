import numpy as np

from mmdet3d.datasets import SemanticKITTIDataset


def test_getitem():
    np.random.seed(0)
    root_path = './tests/data/semantickitti/'
    ann_file = './tests/data/semantickitti/semantickitti_infos.pkl'
    class_names = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
                   'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
                   'sidewalk', 'other-ground', 'building', 'fence',
                   'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign')
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=4,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            seg_3d_dtype=np.int32),
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
                'points',
                'pts_semantic_mask',
            ],
            meta_keys=['file_name', 'sample_idx', 'pcd_rotation']),
    ]

    semantickitti_dataset = SemanticKITTIDataset(root_path, ann_file,
                                                 pipelines)
    data = semantickitti_dataset[0]
    assert data['points']._data.shape[0] == data[
        'pts_semantic_mask']._data.shape[0]
