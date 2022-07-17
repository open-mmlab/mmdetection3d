# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001),
    clip_grad=None)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=150,
        eta_min=0.002,
        by_epoch=True,
        begin=0,
        end=150)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=150)
val_cfg = dict(interval=1)
test_cfg = dict()
