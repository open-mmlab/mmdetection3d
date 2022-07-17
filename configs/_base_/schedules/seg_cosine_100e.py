# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001),
    clip_grad=None)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=100)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=100)
val_cfg = dict(interval=1)
test_cfg = dict()
