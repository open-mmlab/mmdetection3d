# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 1e-4
iter_num_in_epoch = 3712
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=8 * iter_num_in_epoch,
        eta_min=lr * 10,
        by_epoch=False,
        begin=0,
        end=8 * iter_num_in_epoch),
    dict(
        type='CosineAnnealingLR',
        T_max=12 * iter_num_in_epoch,
        eta_min=lr * 1e-4,
        by_epoch=False,
        begin=8 * iter_num_in_epoch,
        end=20 * iter_num_in_epoch),
    dict(
        type='CosineAnnealingBetas',
        T_max=8 * iter_num_in_epoch,
        eta_min=0.85 / 0.95,
        by_epoch=False,
        begin=0,
        end=8 * iter_num_in_epoch),
    dict(
        type='CosineAnnealingBetas',
        T_max=12 * iter_num_in_epoch,
        eta_min=1,
        by_epoch=False,
        begin=8 * iter_num_in_epoch,
        end=20 * iter_num_in_epoch)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20)
val_cfg = dict(interval=1)
test_cfg = dict()
