# dataset settings
dataset_type = 'S3DISDataset'
data_root = './data/s3dis/'
class_names = ('table', 'chair', 'sofa', 'bookcase', 'board')
train_area = [1, 2, 3, 4, 6]
test_area = 5

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointSample', num_points=40000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        # following ScanNet dataset the rotation range is 5 degrees
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
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
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='PointSample', num_points=40000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=data_root + f's3dis_infos_Area_{i}.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=False,
                    classes=class_names,
                    box_type_3d='Depth') for i in train_area
            ],
            separate_eval=False)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

evaluation = dict(pipeline=eval_pipeline)
