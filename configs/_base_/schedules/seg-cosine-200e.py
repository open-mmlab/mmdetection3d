# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.01),
    clip_grad=None)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=200)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
