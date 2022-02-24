# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmdet3d.datasets import NuScenesMonoDataset


def test_getitem():
    np.random.seed(0)
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    img_norm_cfg = dict(
        mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=True,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=1.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]

    nus_dataset = NuScenesMonoDataset(
        ann_file='tests/data/nuscenes/nus_infos_mono3d.coco.json',
        pipeline=pipeline,
        data_root='tests/data/nuscenes/',
        img_prefix='tests/data/nuscenes/',
        test_mode=False)

    data = nus_dataset[0]
    img_metas = data['img_metas']._data
    filename = img_metas['filename']
    img_shape = img_metas['img_shape']
    pad_shape = img_metas['pad_shape']
    flip = img_metas['flip']
    bboxes = data['gt_bboxes']._data
    attrs = data['attr_labels']._data
    labels3d = data['gt_labels_3d']._data
    labels = data['gt_labels']._data
    centers2d = data['centers2d']._data
    depths = data['depths']._data

    expected_filename = 'tests/data/nuscenes/samples/CAM_BACK_LEFT/' + \
        'n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg'
    expected_img_shape = (900, 1600, 3)
    expected_pad_shape = (928, 1600, 3)
    expected_flip = True
    expected_bboxes = torch.tensor([[485.4207, 513.7568, 515.4637, 576.1393],
                                    [748.9482, 512.0452, 776.4941, 571.6310],
                                    [432.1318, 427.8805, 508.4290, 578.1468],
                                    [367.3779, 427.7682, 439.4244, 578.8904],
                                    [592.8713, 515.0040, 623.4984, 575.0945]])
    expected_attr_labels = torch.tensor([8, 8, 4, 4, 8])
    expected_labels = torch.tensor([8, 8, 7, 7, 8])
    expected_centers2d = torch.tensor([[500.6090, 544.6358],
                                       [762.8789, 541.5280],
                                       [471.1633, 502.2295],
                                       [404.1957, 502.5908],
                                       [608.3627, 544.7317]])
    expected_depths = torch.tensor(
        [15.3193, 15.6073, 14.7567, 14.8803, 15.4923])

    assert filename == expected_filename
    assert img_shape == expected_img_shape
    assert pad_shape == expected_pad_shape
    assert flip == expected_flip
    assert torch.allclose(bboxes, expected_bboxes, 1e-5)
    assert torch.all(attrs == expected_attr_labels)
    assert torch.all(labels == expected_labels)
    assert torch.all(labels3d == expected_labels)
    assert torch.allclose(centers2d, expected_centers2d, 1e-5)
    assert torch.allclose(depths, expected_depths, 1e-5)


def test_format_results():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    root_path = 'tests/data/nuscenes/'
    ann_file = 'tests/data/nuscenes/nus_infos_mono3d.coco.json'
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=True,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]
    nus_dataset = NuScenesMonoDataset(
        ann_file=ann_file,
        pipeline=pipeline,
        data_root=root_path,
        test_mode=True)
    results = mmcv.load('tests/data/nuscenes/mono3d_sample_results.pkl')
    result_files, tmp_dir = nus_dataset.format_results(results)
    result_data = mmcv.load(result_files['img_bbox'])
    assert len(result_data['results'].keys()) == 1
    assert len(result_data['results']['e93e98b63d3b40209056d129dc53ceee']) == 8
    det = result_data['results']['e93e98b63d3b40209056d129dc53ceee'][0]

    expected_token = 'e93e98b63d3b40209056d129dc53ceee'
    expected_trans = torch.tensor(
        [1018.753821915645, 605.190386124652, 0.7266818822266328])
    expected_size = torch.tensor([1.440000057220459, 1.6380000114440918, 4.25])
    expected_rotation = torch.tensor([-0.5717, -0.0014, 0.0170, -0.8203])
    expected_detname = 'car'
    expected_attr = 'vehicle.moving'

    assert det['sample_token'] == expected_token
    assert torch.allclose(
        torch.tensor(det['translation']), expected_trans, 1e-5)
    assert torch.allclose(torch.tensor(det['size']), expected_size, 1e-5)
    assert torch.allclose(
        torch.tensor(det['rotation']), expected_rotation, atol=1e-4)
    assert det['detection_name'] == expected_detname
    assert det['attribute_name'] == expected_attr


def test_show():
    root_path = 'tests/data/nuscenes/'
    ann_file = 'tests/data/nuscenes/nus_infos_mono3d.coco.json'
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    eval_pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['img'])
    ]
    nus_dataset = NuScenesMonoDataset(
        data_root=root_path,
        ann_file=ann_file,
        img_prefix='tests/data/nuscenes/',
        test_mode=True,
        pipeline=eval_pipeline)
    results = mmcv.load('tests/data/nuscenes/mono3d_sample_results.pkl')
    results = [results[0]]

    # show with eval_pipeline
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    nus_dataset.show(results, temp_dir, show=False)
    file_name = 'n015-2018-07-18-11-07-57+0800__' \
                'CAM_BACK_LEFT__1531883530447423'
    img_file_path = osp.join(temp_dir, file_name, f'{file_name}_img.png')
    gt_file_path = osp.join(temp_dir, file_name, f'{file_name}_gt.png')
    pred_file_path = osp.join(temp_dir, file_name, f'{file_name}_pred.png')
    mmcv.check_file_exist(img_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()
