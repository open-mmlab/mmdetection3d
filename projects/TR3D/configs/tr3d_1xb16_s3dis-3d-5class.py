_base_ = ['./tr3d.py', 'mmdet3d::_base_/datasets/s3dis-3d.py']
custom_imports = dict(imports=['projects.TR3D.tr3d'])

dataset_type = 'S3DISDataset'
data_root = 'data/s3dis/'
metainfo = dict(classes=('table', 'chair', 'sofa', 'bookcase', 'board'))
train_area = [1, 2, 3, 4, 6]

model = dict(bbox_head=dict(label2level=[1, 0, 1, 1, 0]))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        dataset=dict(datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=f's3dis_infos_Area_{i}.pkl',
                pipeline=train_pipeline,
                filter_empty_gt=False,
                metainfo=metainfo,
                box_type_3d='Depth') for i in train_area
        ])))
