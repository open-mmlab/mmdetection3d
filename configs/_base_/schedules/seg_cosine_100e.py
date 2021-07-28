# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
