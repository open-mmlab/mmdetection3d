import numpy as np
import pytest
import torch

from mmdet3d.datasets import S3DISSegDataset


def test_seg_getitem():
    np.random.seed(0)
    root_path = './tests/data/s3dis/'
    ann_file = './tests/data/s3dis/s3dis_infos.pkl'
    class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                   'door', 'table', 'chair', 'sofa', 'bookcase', 'board',
                   'clutter')
    palette = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
               [255, 0, 255], [100, 100, 255], [200, 200, 100],
               [170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100],
               [200, 200, 200], [50, 50, 50]]
    scene_idxs = [0 for _ in range(20)]

    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            use_color=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True),
        dict(
            type='PointSegClassMapping',
            valid_cat_ids=tuple(range(len(class_names))),
            max_cat_id=13),
        dict(
            type='IndoorPatchPointSample',
            num_points=5,
            block_size=1.0,
            ignore_index=len(class_names),
            use_normalized_coord=True,
            enlarge_size=0.2,
            min_unique_num=None),
        dict(type='NormalizePointsColor', color_mean=None),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=['points', 'pts_semantic_mask'],
            meta_keys=['file_name', 'sample_idx'])
    ]

    s3dis_dataset = S3DISSegDataset(
        data_root=root_path,
        ann_files=ann_file,
        pipeline=pipelines,
        classes=None,
        palette=None,
        modality=None,
        test_mode=False,
        ignore_index=None,
        scene_idxs=scene_idxs)

    data = s3dis_dataset[0]
    points = data['points']._data
    pts_semantic_mask = data['pts_semantic_mask']._data

    file_name = data['img_metas']._data['file_name']
    sample_idx = data['img_metas']._data['sample_idx']

    assert file_name == './tests/data/s3dis/points/Area_1_office_2.bin'
    assert sample_idx == 'Area_1_office_2'
    expected_points = torch.tensor([[
        0.0000, 0.0000, 3.1720, 0.4706, 0.4431, 0.3725, 0.4624, 0.7502, 0.9543
    ], [
        0.2880, -0.5900, 0.0650, 0.3451, 0.3373, 0.3490, 0.5119, 0.5518, 0.0196
    ], [
        0.1570, 0.6000, 3.1700, 0.4941, 0.4667, 0.3569, 0.4893, 0.9519, 0.9537
    ], [
        -0.1320, 0.3950, 0.2720, 0.3216, 0.2863, 0.2275, 0.4397, 0.8830, 0.0818
    ],
                                    [
                                        -0.4860, -0.0640, 3.1710, 0.3843,
                                        0.3725, 0.3059, 0.3789, 0.7286, 0.9540
                                    ]])
    expected_pts_semantic_mask = np.array([0, 1, 0, 8, 0])
    original_classes = s3dis_dataset.CLASSES
    original_palette = s3dis_dataset.PALETTE

    assert s3dis_dataset.CLASSES == class_names
    assert s3dis_dataset.ignore_index == 13
    assert torch.allclose(points, expected_points, 1e-2)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert original_classes == class_names
    assert original_palette == palette
    assert s3dis_dataset.scene_idxs.dtype == np.int32
    assert np.all(s3dis_dataset.scene_idxs == np.array(scene_idxs))

    # test dataset with selected classes
    s3dis_dataset = S3DISSegDataset(
        data_root=root_path,
        ann_files=ann_file,
        pipeline=None,
        classes=['beam', 'window'],
        scene_idxs=scene_idxs)

    label_map = {i: 13 for i in range(14)}
    label_map.update({3: 0, 5: 1})

    assert s3dis_dataset.CLASSES != original_classes
    assert s3dis_dataset.CLASSES == ['beam', 'window']
    assert s3dis_dataset.PALETTE == [palette[3], palette[5]]
    assert s3dis_dataset.VALID_CLASS_IDS == [3, 5]
    assert s3dis_dataset.label_map == label_map
    assert s3dis_dataset.label2cat == {0: 'beam', 1: 'window'}

    # test load classes from file
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('beam\nwindow\n')

    s3dis_dataset = S3DISSegDataset(
        data_root=root_path,
        ann_files=ann_file,
        pipeline=None,
        classes=tmp_file.name,
        scene_idxs=scene_idxs)
    assert s3dis_dataset.CLASSES != original_classes
    assert s3dis_dataset.CLASSES == ['beam', 'window']
    assert s3dis_dataset.PALETTE == [palette[3], palette[5]]
    assert s3dis_dataset.VALID_CLASS_IDS == [3, 5]
    assert s3dis_dataset.label_map == label_map
    assert s3dis_dataset.label2cat == {0: 'beam', 1: 'window'}

    # test scene_idxs in dataset
    # we should input scene_idxs in train mode
    with pytest.raises(NotImplementedError):
        s3dis_dataset = S3DISSegDataset(
            data_root=root_path,
            ann_files=ann_file,
            pipeline=None,
            scene_idxs=None)

    # test mode
    s3dis_dataset = S3DISSegDataset(
        data_root=root_path,
        ann_files=ann_file,
        pipeline=None,
        test_mode=True,
        scene_idxs=scene_idxs)
    assert np.all(s3dis_dataset.scene_idxs == np.array([0]))


def test_seg_evaluate():
    if not torch.cuda.is_available():
        pytest.skip()
    root_path = './tests/data/s3dis'
    ann_file = './tests/data/s3dis/s3dis_infos.pkl'
    s3dis_dataset = S3DISSegDataset(
        data_root=root_path, ann_files=ann_file, test_mode=True)
    results = []
    pred_sem_mask = dict(
        semantic_mask=torch.tensor([
            2, 3, 1, 2, 2, 6, 1, 0, 1, 1, 9, 12, 3, 0, 2, 0, 2, 0, 8, 3, 1, 2,
            0, 2, 1, 7, 2, 10, 2, 0, 0, 0, 2, 3, 2, 2, 2, 2, 2, 3, 0, 0, 4, 6,
            7, 2, 1, 2, 0, 1, 7, 0, 2, 2, 2, 0, 2, 2, 1, 12, 0, 2, 2, 2, 2, 7,
            2, 2, 0, 2, 6, 2, 12, 6, 3, 12, 2, 1, 6, 1, 2, 6, 8, 2, 10, 1, 11,
            0, 6, 9, 4, 3, 0, 0, 12, 1, 1, 5, 3, 2
        ]).long())
    results.append(pred_sem_mask)
    ret_dict = s3dis_dataset.evaluate(results)
    assert abs(ret_dict['miou'] - 0.7625) < 0.01
    assert abs(ret_dict['acc'] - 0.9) < 0.01
    assert abs(ret_dict['acc_cls'] - 0.9074) < 0.01


def test_seg_show():
    import mmcv
    import tempfile
    from os import path as osp

    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    root_path = './tests/data/s3dis'
    ann_file = './tests/data/s3dis/s3dis_infos.pkl'
    s3dis_dataset = S3DISSegDataset(
        data_root=root_path, ann_files=ann_file, scene_idxs=[0])
    result = dict(
        semantic_mask=torch.tensor([
            2, 2, 1, 2, 2, 5, 1, 0, 1, 1, 9, 12, 3, 0, 2, 0, 2, 0, 8, 2, 0, 2,
            0, 2, 1, 7, 2, 10, 2, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 0, 4, 6,
            7, 2, 1, 2, 0, 1, 7, 0, 2, 2, 2, 0, 2, 2, 1, 12, 0, 2, 2, 2, 2, 7,
            2, 2, 0, 2, 6, 2, 12, 6, 2, 12, 2, 1, 6, 1, 2, 6, 8, 2, 10, 1, 10,
            0, 6, 9, 4, 3, 0, 0, 12, 1, 1, 5, 2, 2
        ]).long())
    results = [result]
    s3dis_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, 'Area_1_office_2',
                             'Area_1_office_2_points.obj')
    gt_file_path = osp.join(temp_dir, 'Area_1_office_2',
                            'Area_1_office_2_gt.obj')
    pred_file_path = osp.join(temp_dir, 'Area_1_office_2',
                              'Area_1_office_2_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()
    # test show with pipeline
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                   'door', 'table', 'chair', 'sofa', 'bookcase', 'board',
                   'clutter')
    eval_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            use_color=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True),
        dict(
            type='PointSegClassMapping',
            valid_cat_ids=tuple(range(len(class_names))),
            max_cat_id=13),
        dict(
            type='DefaultFormatBundle3D',
            with_label=False,
            class_names=class_names),
        dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
    ]
    s3dis_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, 'Area_1_office_2',
                             'Area_1_office_2_points.obj')
    gt_file_path = osp.join(temp_dir, 'Area_1_office_2',
                            'Area_1_office_2_gt.obj')
    pred_file_path = osp.join(temp_dir, 'Area_1_office_2',
                              'Area_1_office_2_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()


def test_multi_areas():
    # S3DIS dataset has 6 areas, we often train on several of them
    # need to verify the concat function of S3DISSegDataset
    root_path = './tests/data/s3dis'
    ann_file = './tests/data/s3dis/s3dis_infos.pkl'
    class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                   'door', 'table', 'chair', 'sofa', 'bookcase', 'board',
                   'clutter')
    palette = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
               [255, 0, 255], [100, 100, 255], [200, 200, 100],
               [170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100],
               [200, 200, 200], [50, 50, 50]]
    scene_idxs = [0 for _ in range(20)]

    # repeat
    repeat_num = 3
    s3dis_dataset = S3DISSegDataset(
        data_root=root_path,
        ann_files=[ann_file for _ in range(repeat_num)],
        scene_idxs=scene_idxs)
    assert s3dis_dataset.CLASSES == class_names
    assert s3dis_dataset.PALETTE == palette
    assert len(s3dis_dataset.data_infos) == repeat_num
    assert np.all(s3dis_dataset.scene_idxs == np.concatenate(
        [np.array(scene_idxs) + i for i in range(repeat_num)]))

    # different scene_idxs input
    s3dis_dataset = S3DISSegDataset(
        data_root=root_path,
        ann_files=[ann_file for _ in range(repeat_num)],
        scene_idxs=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 3, 4], [0, 1, 1, 2, 2, 2]])
    assert np.all(s3dis_dataset.scene_idxs == np.array(
        [0, 0, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 10]))
