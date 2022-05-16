# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
import torch

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
            type='LoadCalibPoseTime',
            use_calib=True,
            use_pose=True,
            use_time=True,
        ),
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


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    data_root = './tests/data/semantickitti/'
    ann_file = './tests/data/semantickitti/semantickitti_infos.pkl'
    classes = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign')
    learning_map = {
        0: 0,  # "unlabeled"
        1: 0,  # "outlier" mapped to "unlabeled" ---------------mapped
        10: 1,  # "car"
        11: 2,  # "bicycle"
        13: 5,  # "bus" mapped to "other-vehicle" ---------------mapped
        15: 3,  # "motorcycle"
        16: 5,  # "on-rails" mapped to "other-vehicle" ----------mapped
        18: 4,  # "truck"
        20: 5,  # "other-vehicle"
        30: 6,  # "person"
        31: 7,  # "bicyclist"
        32: 8,  # "motorcyclist"
        40: 9,  # "road"
        44: 10,  # "parking"
        48: 11,  # "sidewalk"
        49: 12,  # "other-ground"
        50: 13,  # "building"
        51: 14,  # "fence"
        52: 0,  # "other-structure" mapped to "unlabeled" -------mapped
        60: 9,  # "lane-marking" to "road" ----------------------mapped
        70: 15,  # "vegetation"
        71: 16,  # "trunk"
        72: 17,  # "terrain"
        80: 18,  # "pole"
        81: 19,  # "traffic-sign"
        99: 0,  # "other-object" to "unlabeled" -----------------mapped
        252: 1,  # "moving-car" to "car" -------------------------mapped
        253: 7,  # "moving-bicyclist" to "bicyclist" -------------mapped
        254: 6,  # "moving-person" to "person" -------------------mapped
        255: 8,  # "moving-motorcyclist" to "motorcyclist" -------mapped
        256: 5,  # "moving-on-rails" mapped to "other-vehicle" ---mapped
        257: 5,  # "moving-bus" mapped to "other-vehicle" --------mapped
        258: 4,  # "moving-truck" to "truck" ---------------------mapped
        259: 5  # "moving-other"-vehicle to "other-vehicle" -----mapped
    }
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
        dict(type='DefaultFormatBundle3D', class_names=classes),
        dict(
            type='Collect3D',
            keys=[
                'points',
                'pts_semantic_mask',
            ],
            meta_keys=[
                'file_name', 'sample_idx', 'pcd_rotation', 'calib', 'pose'
            ]),
    ]

    ignore = [0]
    min_points = 1
    semantic_kitti_dataset = SemanticKITTIDataset(
        data_root,
        ann_file,
        pipeline=pipelines,
        classes=classes,
        ignore=ignore,
        min_points=min_points)

    class_remap = learning_map
    inst_pred_0 = np.fromfile(
        osp.join(data_root, 'sequences/00/labels/000000.label'),
        dtype=np.int32).reshape(-1, 1)
    sem_pred_0 = inst_pred_0 & 0xFFFF
    sem_pred_0 = np.vectorize(class_remap.__getitem__)(sem_pred_0)

    result0 = dict(pred_sem=sem_pred_0, pred_inst=inst_pred_0)

    ap_dict = semantic_kitti_dataset.evaluate([result0])

    assert np.isclose(ap_dict[0]['all']['PQ'], 0.21052631578947367)
    assert np.isclose(ap_dict[0]['all']['SQ'], 0.21052631578947367)
    assert np.isclose(ap_dict[0]['all']['RQ'], 0.21052631578947367)
