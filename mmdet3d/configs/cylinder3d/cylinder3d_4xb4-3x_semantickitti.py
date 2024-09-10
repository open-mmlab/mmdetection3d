# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.semantickitti import *
    from .._base_.models.cylinder3d import *
    from .._base_.default_runtime import *

from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim import AdamW

# optimizer
lr = 0.001
optim_wrapper = dict(
    type=OptimWrapper, optimizer=dict(type=AdamW, lr=lr, weight_decay=0.01))

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=36, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

train_dataloader.update(dict(batch_size=4, ))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks.update(dict(checkpoint=dict(type=CheckpointHook, interval=5)))
