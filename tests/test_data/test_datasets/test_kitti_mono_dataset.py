import mmcv
import numpy as np
import pytest
import torch

from mmdet3d.datasets import KittiMonoDataset


def test_getitem():
    np.random.seed(0)
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=False,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=1.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]

    kitti_dataset = KittiMonoDataset(
        ann_file='tests/data/kitti/kitti_infos_mono3d.coco.json',
        info_file='tests/data/kitti/kitti_infos_mono3d.pkl',
        pipeline=pipeline,
        data_root='tests/data/kitti/',
        img_prefix='tests/data/kitti/',
        test_mode=False)

    data = kitti_dataset[0]
    img_metas = data['img_metas']._data
    filename = img_metas['filename']
    img_shape = img_metas['img_shape']
    pad_shape = img_metas['pad_shape']
    flip = img_metas['flip']
    bboxes = data['gt_bboxes']._data
    labels3d = data['gt_labels_3d']._data
    labels = data['gt_labels']._data
    centers2d = data['centers2d']._data
    depths = data['depths']._data

    expected_filename = 'tests/data/kitti/training/image_2/000007.png'
    expected_img_shape = (375, 1242, 3)
    expected_pad_shape = (384, 1248, 3)
    expected_flip = True
    expected_bboxes = torch.tensor([[625.3445, 175.0120, 676.5177, 224.9605],
                                    [729.5906, 179.8571, 760.1503, 202.5390],
                                    [676.7557, 175.7334, 699.7753, 193.9447],
                                    [886.5021, 176.1380, 911.1581, 213.8148]])
    expected_labels = torch.tensor([2, 2, 2, 1])
    expected_centers2d = torch.tensor([[650.6185, 198.3731],
                                       [744.2711, 190.7532],
                                       [687.8787, 184.5331],
                                       [898.4750, 194.4337]])
    expected_depths = torch.tensor([25.0127, 47.5527, 60.5227, 34.0927])

    assert filename == expected_filename
    assert img_shape == expected_img_shape
    assert pad_shape == expected_pad_shape
    assert flip == expected_flip
    assert torch.allclose(bboxes, expected_bboxes, 1e-5)
    assert torch.all(labels == expected_labels)
    assert torch.all(labels3d == expected_labels)
    assert torch.allclose(centers2d, expected_centers2d, 1e-5)
    assert torch.allclose(depths, expected_depths, 1e-5)


def test_format_results():
    root_path = 'tests/data/kitti/'
    info_file = 'tests/data/kitti/kitti_infos_mono3d.pkl'
    ann_file = 'tests/data/kitti/kitti_infos_mono3d.coco.json'
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=False,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]
    kitti_dataset = KittiMonoDataset(
        ann_file=ann_file,
        info_file=info_file,
        pipeline=pipeline,
        data_root=root_path,
        test_mode=True)

    # format 3D detection results
    results = mmcv.load('tests/data/kitti/mono3d_sample_results.pkl')
    result_files, tmp_dir = kitti_dataset.format_results(results)
    result_data = result_files['img_bbox']
    assert len(result_data) == 1
    assert len(result_data[0]['name']) == 4
    det = result_data[0]

    expected_bbox = torch.tensor([[565.4989, 175.02547, 616.70184, 225.00565],
                                  [481.85907, 179.8642, 512.43414, 202.5624],
                                  [542.23157, 175.73912, 565.26263, 193.96303],
                                  [330.8572, 176.1482, 355.53937, 213.8469]])
    expected_dims = torch.tensor([[3.201, 1.6110001, 1.661],
                                  [3.701, 1.401, 1.511],
                                  [4.051, 1.4610001, 1.661],
                                  [1.9510001, 1.7210001, 0.501]])
    expected_rotation = torch.tensor([-1.59, 1.55, 1.56, 1.54])
    expected_detname = ['Car', 'Car', 'Car', 'Cyclist']

    assert torch.allclose(torch.from_numpy(det['bbox']), expected_bbox, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['dimensions']), expected_dims, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['rotation_y']), expected_rotation, 1e-5)
    assert det['name'].tolist() == expected_detname

    # format 2D detection results
    results = mmcv.load('tests/data/kitti/mono3d_sample_results2d.pkl')
    result_files, tmp_dir = kitti_dataset.format_results(results)
    result_data = result_files['img_bbox2d']
    assert len(result_data) == 1
    assert len(result_data[0]['name']) == 4
    det = result_data[0]

    expected_bbox = torch.tensor(
        [[330.84191493, 176.13804312, 355.49885373, 213.81578769],
         [565.48227204, 175.01202566, 616.65650883, 224.96147091],
         [481.84967085, 179.85710612, 512.41043776, 202.54001526],
         [542.22471517, 175.73341152, 565.24534908, 193.94568878]])
    expected_dims = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                                  [0., 0., 0.]])
    expected_rotation = torch.tensor([0., 0., 0., 0.])
    expected_detname = ['Cyclist', 'Car', 'Car', 'Car']

    assert torch.allclose(
        torch.from_numpy(det['bbox']).float(), expected_bbox, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['dimensions']).float(), expected_dims, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['rotation_y']).float(), expected_rotation, 1e-5)
    assert det['name'].tolist() == expected_detname


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    root_path = 'tests/data/kitti/'
    info_file = 'tests/data/kitti/kitti_infos_mono3d.pkl'
    ann_file = 'tests/data/kitti/kitti_infos_mono3d.coco.json'
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=False,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]
    kitti_dataset = KittiMonoDataset(
        ann_file=ann_file,
        info_file=info_file,
        pipeline=pipeline,
        data_root=root_path,
        test_mode=True)

    # format 3D detection results
    results = mmcv.load('tests/data/kitti/mono3d_sample_results.pkl')
    results2d = mmcv.load('tests/data/kitti/mono3d_sample_results2d.pkl')
    results[0]['img_bbox2d'] = results2d[0]['img_bbox2d']

    metric = ['mAP']
    ap_dict = kitti_dataset.evaluate(results, metric)
    assert np.isclose(ap_dict['img_bbox/KITTI/Overall_3D_easy'], 3.0303)
    assert np.isclose(ap_dict['img_bbox/KITTI/Overall_3D_moderate'], 6.0606)
    assert np.isclose(ap_dict['img_bbox/KITTI/Overall_3D_hard'], 6.0606)
    assert np.isclose(ap_dict['img_bbox2d/KITTI/Overall_2D_easy'], 3.0303)
    assert np.isclose(ap_dict['img_bbox2d/KITTI/Overall_2D_moderate'], 6.0606)
    assert np.isclose(ap_dict['img_bbox2d/KITTI/Overall_2D_hard'], 6.0606)
