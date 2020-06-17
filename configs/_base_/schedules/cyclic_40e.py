# The schedule is usually used by models trained on KITTI dataset

# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4
lr = 0.0018
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the offcial AdamW optimizer implemented by PyTorch.
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# We use cyclic learning rate and momentum schedule following SECOND.Pytorch
# https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/torchplus/train/learning_schedules_fastai.py#L69  # noqa
# We implement them in mmcv, for more details, please refer to
# https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327  # noqa
# https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130  # noqa
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
# Although the total_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual total epoch
# number could be Nx40
total_epochs = 40
