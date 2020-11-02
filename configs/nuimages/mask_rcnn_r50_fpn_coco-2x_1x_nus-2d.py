_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/nuim_instance.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10), mask_head=dict(num_classes=10)))

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/nuscenes/': 's3://nuscenes/nuscenes/',
        'data/nuscenes/': 's3://nuscenes/nuscenes/'
    }))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = 'data/nuimages/'
# data = dict(
#     val=dict(
#         ann_file=data_root + 'annotations/nuimages_v1.0-mini.json'),
#     test=dict(
#         ann_file=data_root + 'annotations/nuimages_v1.0-mini.json'))
