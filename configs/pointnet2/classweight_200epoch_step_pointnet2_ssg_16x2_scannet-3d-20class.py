_base_ = [
    '../_base_/datasets/scannet_seg-3d-20class.py',
    '../_base_/models/pointnet2_ssg.py', '../_base_/default_runtime.py'
]

# data settings
data_root = '/mnt/lustre/wuziyi/Data_t1/ScanNetV2/'
data = dict(samples_per_gpu=16)
evaluation = dict(interval=5)

# model settings
model = dict(
    decode_head=dict(
        num_classes=20,
        ignore_index=20,
        loss_decode=dict(class_weight=data_root +
                         'seg_info/train_label_weight.npy')),
    test_cfg=dict(
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=32))

# optimizer
lr = 0.001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, gamma=0.7, step=10, min_lr=1e-5)
# lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

# runtime settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=200)
dist_params = dict(port=29503)
