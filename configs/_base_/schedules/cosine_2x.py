lr = 1e-5

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        betas=(0.9, 0.999),  # the momentum is change during training
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    clip_grad=dict(grad_clip=dict(max_norm=10, norm_type=2))
)

lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')