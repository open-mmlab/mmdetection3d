_base_ = [
    '../_base_/models/parta2.py', '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=2, num_workers=2)

# Part-A2 uses a different learning rate from what SECOND uses.
optim_wrapper = dict(optimizer=dict(lr=0.001))
find_unused_parameters = True

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
