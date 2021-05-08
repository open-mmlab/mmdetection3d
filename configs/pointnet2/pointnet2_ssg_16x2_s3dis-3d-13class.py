_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py',
    '../_base_/models/pointnet2_ssg.py', '../_base_/default_runtime.py'
]

# data settings
data_root = './data/s3dis/'
data = dict(samples_per_gpu=16)
evaluation = dict(interval=5)

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

# optimizer
lr = 0.001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

# runtime settings
checkpoint_config = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=150)
