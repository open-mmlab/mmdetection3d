# The schedule is usually used by models trained on KITTI dataset
# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4
lr = 0.0018
iter_num_in_epoch = 3712
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the official AdamW optimizer implemented by PyTorch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=16 * iter_num_in_epoch,
        eta_min=lr * 10,
        by_epoch=False,
        begin=0,
        end=16 * iter_num_in_epoch),
    dict(
        type='CosineAnnealingLR',
        T_max=24 * iter_num_in_epoch,
        eta_min=lr * 1e-4,
        by_epoch=False,
        begin=16 * iter_num_in_epoch,
        end=40 * iter_num_in_epoch),
    dict(
        type='CosineAnnealingBetas',
        T_max=16 * iter_num_in_epoch,
        eta_min=0.85 / 0.95,
        by_epoch=False,
        begin=0,
        end=16 * iter_num_in_epoch),
    dict(
        type='CosineAnnealingBetas',
        T_max=24 * iter_num_in_epoch,
        eta_min=1,
        by_epoch=False,
        begin=16 * iter_num_in_epoch,
        end=40 * iter_num_in_epoch)
]

# Runtime settings，training schedule for 40e
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
train_cfg = dict(by_epoch=True, max_epochs=40)
val_cfg = dict(interval=1)
test_cfg = dict()
