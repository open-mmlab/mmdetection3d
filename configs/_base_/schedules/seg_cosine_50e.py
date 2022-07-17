# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.001),
    clip_grad=None)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=50)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = dict(interval=1)
test_cfg = dict()
