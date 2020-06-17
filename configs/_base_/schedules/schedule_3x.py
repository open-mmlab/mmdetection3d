# optimizer
lr = 0.008  # max learning rate
optimizer = dict(type='Adam', lr=lr)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32])
# runtime settings
total_epochs = 36
