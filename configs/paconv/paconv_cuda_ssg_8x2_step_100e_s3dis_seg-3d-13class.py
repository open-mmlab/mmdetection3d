_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py',
    '../_base_/models/paconv_cuda_ssg.py', '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=8)
evaluation = dict(interval=2)

# model settings
model = dict(
    decode_head=dict(
        num_classes=13, ignore_index=13,
        loss_decode=dict(class_weight=None)),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=12))

# runtime settings
checkpoint_config = dict(interval=2)

# optimizer
optimizer = dict(type='SGD', lr=0.05, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='StepLrUpdaterHook', warmup=None, step=[60, 80], gamma=0.1)
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
