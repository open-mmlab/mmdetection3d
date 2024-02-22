# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import OneCycleLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.adamw import AdamW

# training schedule for 80e
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=80, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
lr = 0.01
param_scheduler = [
    dict(
        type=OneCycleLR,
        eta_max=lr,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
        by_epoch=False)
]

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
