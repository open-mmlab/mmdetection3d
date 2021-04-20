# dataset settings
dataset_type = 'S3DISSegDataset'
data_root = './data/s3dis/'
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
num_points = 4096
train_area = [1, 2, 3, 4, 6]
test_area = 5
train_pipeline = [
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
        valid_cat_ids=tuple(range(len(class_names)))),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.0,
        sample_rate=1.0,
        ignore_index=len(class_names),
        use_normalized_coord=True),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
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
        valid_cat_ids=tuple(range(len(class_names)))),
    dict(
        type='DefaultFormatBundle3D',
        with_label=False,
        class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    # train on area 1, 2, 3, 4, 6
    # test on area 5
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[
            data_root + f's3dis_infos_Area_{i}.pkl' for i in train_area
        ],
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        ignore_index=len(class_names),
        scene_idxs=[
            data_root + f'seg_info/Area_{i}_resampled_scene_idxs.npy'
            for i in train_area
        ],
        label_weight=[
            data_root + f'seg_info/Area_{i}_label_weight.npy'
            for i in train_area
        ]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names)),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names)))

evaluation = dict(pipeline=eval_pipeline)
