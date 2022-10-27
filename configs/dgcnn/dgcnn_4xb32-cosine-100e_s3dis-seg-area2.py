_base_ = [
    '../_base_/datasets/s3dis-seg.py', '../_base_/models/dgcnn.py',
    '../_base_/schedules/seg-cosine-100e.py', '../_base_/default_runtime.py'
]

# data settings
train_area = [1, 3, 4, 5, 6]
test_area = 2
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_files=[f's3dis_infos_Area_{i}.pkl' for i in train_area],
        scene_idxs=[
            f'seg_info/Area_{i}_resampled_scene_idxs.npy' for i in train_area
        ]))
test_dataloader = dict(
    dataset=dict(
        ann_files=f's3dis_infos_Area_{test_area}.pkl',
        scene_idxs=f'seg_info/Area_{test_area}_resampled_scene_idxs.npy'))
val_dataloader = test_dataloader

# model settings
model = dict(
    backbone=dict(in_channels=9),  # [xyz, rgb, normalized_xyz]
    decode_head=dict(
        num_classes=13, ignore_index=13,
        loss_decode=dict(class_weight=None)),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=24))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
train_cfg = dict(val_interval=2)
