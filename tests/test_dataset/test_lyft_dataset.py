import mmcv
import numpy as np
import torch

from mmdet3d.datasets import LyftDataset


def test_getitem():
    np.random.seed(0)
    torch.manual_seed(0)
    root_path = './tests/data/lyft'
    ann_file = './tests/data/lyft/lyft_infos.pkl'
    class_names = ('car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
                   'motorcycle', 'bicycle', 'pedestrian', 'animal')
    point_cloud_range = [-80, -80, -10, 80, 80, 10]
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=2,
            file_client_args=dict(backend='disk')),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.523599, 0.523599],
            scale_ratio_range=[0.85, 1.15],
            translation_std=[0, 0, 0]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    lyft_dataset = LyftDataset(ann_file, pipelines, root_path)
    data = lyft_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels_3d = data['gt_labels_3d']._data
    pts_filename = data['img_metas']._data['pts_filename']
    pcd_horizontal_flip = data['img_metas']._data['pcd_horizontal_flip']
    pcd_scale_factor = data['img_metas']._data['pcd_scale_factor']
    pcd_rotation = data['img_metas']._data['pcd_rotation']
    sample_idx = data['img_metas']._data['sample_idx']
    pcd_rotation_expected = np.array([[0.99869376, -0.05109515, 0.],
                                      [0.05109515, 0.99869376, 0.],
                                      [0., 0., 1.]])
    assert pts_filename == \
        'tests/data/lyft/lidar/host-a017_lidar1_1236118886901125926.bin'
    assert pcd_horizontal_flip is True
    assert abs(pcd_scale_factor - 1.0645568099117257) < 1e-5
    assert np.allclose(pcd_rotation, pcd_rotation_expected, 1e-3)
    assert sample_idx == \
        'b98a05255ba2632e957884758cb31f0e6fcc8d3cd6ee76b6d0ba55b72f08fc54'
    expected_points = torch.tensor([[61.4785, -3.7393, 6.7699, 0.4001],
                                    [47.7904, -3.9887, 6.0926, 0.0000],
                                    [52.5683, -4.2178, 6.7179, 0.0000],
                                    [52.4867, -4.0315, 6.7057, 0.0000],
                                    [59.8372, -1.7366, 6.5864, 0.4001],
                                    [53.0842, -3.7064, 6.7811, 0.0000],
                                    [60.5549, -3.4978, 6.6578, 0.4001],
                                    [59.1695, -1.2910, 7.0296, 0.2000],
                                    [53.0702, -3.8868, 6.7807, 0.0000],
                                    [47.9579, -4.1648, 5.6219, 0.2000],
                                    [59.8226, -1.5522, 6.5867, 0.4001],
                                    [61.2858, -4.2254, 7.3089, 0.2000],
                                    [49.9896, -4.5202, 5.8823, 0.2000],
                                    [61.4597, -4.6402, 7.3340, 0.2000],
                                    [59.8244, -1.3499, 6.5895, 0.4001]])
    expected_gt_bboxes_3d = torch.tensor(
        [[63.2257, 17.5206, -0.6307, 2.0109, 5.1652, 1.9471, -1.5868],
         [-25.3804, 27.4598, -2.3297, 2.7412, 8.4792, 3.4343, -1.5939],
         [-15.2098, -7.0109, -2.2566, 0.7931, 0.8410, 1.7916, 1.5090]])
    expected_gt_labels = np.array([0, 4, 7])
    original_classes = lyft_dataset.CLASSES

    assert torch.allclose(points, expected_points, 1e-2)
    assert torch.allclose(gt_bboxes_3d.tensor, expected_gt_bboxes_3d, 1e-3)
    assert np.all(gt_labels_3d.numpy() == expected_gt_labels)
    assert original_classes == class_names

    lyft_dataset = LyftDataset(
        ann_file, None, root_path, classes=['car', 'pedestrian'])
    assert lyft_dataset.CLASSES != original_classes
    assert lyft_dataset.CLASSES == ['car', 'pedestrian']

    lyft_dataset = LyftDataset(
        ann_file, None, root_path, classes=('car', 'pedestrian'))
    assert lyft_dataset.CLASSES != original_classes
    assert lyft_dataset.CLASSES == ('car', 'pedestrian')

    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('car\npedestrian\n')

    lyft_dataset = LyftDataset(
        ann_file, None, root_path, classes=tmp_file.name)
    assert lyft_dataset.CLASSES != original_classes
    assert lyft_dataset.CLASSES == ['car', 'pedestrian']


def test_evaluate():
    root_path = './tests/data/lyft'
    ann_file = './tests/data/lyft/lyft_infos_val.pkl'
    lyft_dataset = LyftDataset(ann_file, None, root_path)
    results = mmcv.load('./tests/data/lyft/sample_results.pkl')
    ap_dict = lyft_dataset.evaluate(results, 'bbox')
    car_precision = ap_dict['pts_bbox_Lyft/car_AP']
    assert car_precision == 0.6
