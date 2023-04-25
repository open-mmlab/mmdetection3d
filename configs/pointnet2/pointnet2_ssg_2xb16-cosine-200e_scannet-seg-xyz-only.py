_base_ = [
    '../_base_/datasets/scannet-seg.py', '../_base_/models/pointnet2_ssg.py',
    '../_base_/schedules/seg-cosine-200e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(in_channels=3),  # only [xyz]
    decode_head=dict(
        num_classes=20,
        ignore_index=20,
        # `class_weight` is generated in data pre-processing, saved in
        # `data/scannet/seg_info/train_label_weight.npy`
        # you can copy paste the values here, or input the file path as
        # `class_weight=data/scannet/seg_info/train_label_weight.npy`
        loss_decode=dict(class_weight=[
            2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
            4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
            5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
            5.3954206, 4.6971426
        ])),
    test_cfg=dict(
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))

# dataset settings
# in this setting, we only use xyz as network input
# so we need to re-write all the data pipeline
class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')
num_points = 8192
backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2],  # only load xyz coordinates
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.5,
        ignore_index=len(class_names),
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(batch_size=16, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
train_cfg = dict(val_interval=5)
