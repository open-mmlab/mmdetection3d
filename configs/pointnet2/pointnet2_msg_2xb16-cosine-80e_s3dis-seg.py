_base_ = [
    '../_base_/datasets/s3dis-seg.py', '../_base_/models/pointnet2_msg.py',
    '../_base_/schedules/seg-cosine-50e.py', '../_base_/default_runtime.py'
]

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

# data settings
train_dataloader = dict(batch_size=16)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)
