_base_ = ['./tr3d.py', 'mmdet3d::_base_/datasets/sunrgbd-3d.py']
custom_imports = dict(imports=['projects.TR3D.tr3d'])

model = dict(
    bbox_head=dict(
        num_reg_outs=8,
        label2level=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        bbox_loss=dict(
            type='TR3DRotatedIoU3DLoss', mode='diou', reduction='none')))

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
        flip_ratio_bev_vertical=0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample', num_points=100000),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(pipeline=train_pipeline, filter_empty_gt=False)))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
