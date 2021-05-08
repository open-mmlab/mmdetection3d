_base_ = [
    '../_base_/datasets/scannet_seg-3d-20class.py',
    '../_base_/models/pointnet2_msg.py', '../_base_/default_runtime.py'
]

# data settings
data_root = './data/scannet/'
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
        batch_size=24))

# optimizer
lr = 0.001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

# runtime settings
checkpoint_config = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=150)
