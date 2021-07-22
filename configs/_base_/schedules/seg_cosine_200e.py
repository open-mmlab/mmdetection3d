# optimizer
# This schedule is mainly used on ScanNet dataset in segmentation task
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
