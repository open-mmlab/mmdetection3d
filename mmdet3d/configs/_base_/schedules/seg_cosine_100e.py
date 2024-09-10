# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD

# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.1, momentum=0.9, weight_decay=0.001),
    clip_grad=None)

param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        T_max=100,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=100)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
