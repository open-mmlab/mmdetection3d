# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import mmcv
import os.path as osp
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
# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile

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
            type='LoadCalibPoseTime',
            use_calib=True,
            use_pose=True,
            use_time=True,
            ),
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
            meta_keys=['file_name', 'sample_idx', 'pcd_rotation', 'calib', 'pose']),
    ]

    ignore = [0]
    min_points = 1
    global_cfg = mmcv.load(osp.join(data_root, 'semantic-kitti.yaml'))
    class_remap = global_cfg["learning_map"]

    semantic_kitti_dataset = SemanticKITTIDataset(data_root, ann_file, pipeline=pipelines, classes=classes, ignore=ignore, min_points=min_points)

    inst_pred_0 = np.fromfile(osp.join(data_root, 'sequences/00/labels/000000.label'), dtype=np.int32).reshape(-1, 1)
    sem_pred_0 = inst_pred_0 & 0xFFFF
    sem_pred_0 = np.vectorize(class_remap.__getitem__)(sem_pred_0)

    # sem_pred_0, inst_pred_0, sem_gt_0, inst_gt_0 = gen_psuedo_labels(50)
    # sem_pred_1, inst_pred_1, sem_gt_1, inst_gt_1 = gen_psuedo_labels(51)
    result0 = dict(pred_sem=sem_pred_0, pred_inst=inst_pred_0)

    ap_dict = semantic_kitti_dataset.evaluate([result0])

    assert np.isclose(ap_dict[0]["all"]["PQ"], 0.21052631578947367)
    assert np.isclose(ap_dict[0]["all"]["SQ"], 0.21052631578947367)
    assert np.isclose(ap_dict[0]["all"]["RQ"], 0.21052631578947367)


def gen_psuedo_labels(n=50):
    # generate ground truth and prediction
    sem_pred = []
    inst_pred = []
    sem_gt = []
    inst_gt = []

    # some ignore stuff
    N_ignore = n
    sem_pred.extend([0 for i in range(N_ignore)])
    inst_pred.extend([0 for i in range(N_ignore)])
    sem_gt.extend([0 for i in range(N_ignore)])
    inst_gt.extend([0 for i in range(N_ignore)])

    # grass segment
    N_grass = n+1
    N_grass_pred = np.random.randint(0, N_grass)  # rest is sky
    sem_pred.extend([1 for i in range(N_grass_pred)])  # grass
    sem_pred.extend([2 for i in range(N_grass - N_grass_pred)])  # sky
    inst_pred.extend([0 for i in range(N_grass)])
    sem_gt.extend([1 for i in range(N_grass)])  # grass
    inst_gt.extend([0 for i in range(N_grass)])

    # sky segment
    N_sky = n+2
    N_sky_pred = np.random.randint(0, N_sky)  # rest is grass
    sem_pred.extend([2 for i in range(N_sky_pred)])  # sky
    sem_pred.extend([1 for i in range(N_sky - N_sky_pred)])  # grass
    inst_pred.extend([0 for i in range(N_sky)])  # first instance
    sem_gt.extend([2 for i in range(N_sky)])  # sky
    inst_gt.extend([0 for i in range(N_sky)])  # first instance

    # wrong dog as person prediction
    N_dog = n+3
    N_person = N_dog
    sem_pred.extend([3 for i in range(N_person)])
    inst_pred.extend([35 for i in range(N_person)])
    sem_gt.extend([4 for i in range(N_dog)])
    inst_gt.extend([22 for i in range(N_dog)])

    # two persons in prediction, but three in gt
    N_person = n+4
    sem_pred.extend([3 for i in range(6 * N_person)])
    inst_pred.extend([8 for i in range(4 * N_person)])
    inst_pred.extend([95 for i in range(2 * N_person)])
    sem_gt.extend([3 for i in range(6 * N_person)])
    inst_gt.extend([33 for i in range(3 * N_person)])
    inst_gt.extend([42 for i in range(N_person)])
    inst_gt.extend([11 for i in range(2 * N_person)])

    # gt and pred to numpy
    sem_pred = np.array(sem_pred, dtype=np.int64).reshape(1, -1)
    inst_pred = np.array(inst_pred, dtype=np.int64).reshape(1, -1)
    sem_gt = np.array(sem_gt, dtype=np.int64).reshape(1, -1)
    inst_gt = np.array(inst_gt, dtype=np.int64).reshape(1, -1)

    return sem_pred, inst_pred, sem_gt, inst_gt


if __name__ == "__main__":
    test_getitem()
    test_evaluate()
