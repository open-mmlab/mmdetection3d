dataset_type = 'CocoDataset'
data_root = 'data/nuimages/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://nuimages'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1280, 720), (1920, 1080)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/nuimages_v1.0-train.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/nuimages_v1.0-val.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/nuimages_v1.0-val.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
