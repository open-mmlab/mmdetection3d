# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 0.003  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, weight_decay=0.001, betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=40,
        end=40,
        by_epoch=True,
        eta_min=1e-5)
]
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
