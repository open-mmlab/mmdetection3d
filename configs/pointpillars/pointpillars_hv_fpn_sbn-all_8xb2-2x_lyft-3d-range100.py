_base_ = [
    '../_base_/models/pointpillars_hv_fpn_range100_lyft.py',
    '../_base_/datasets/lyft-3d-range100.py',
    '../_base_/schedules/schedule-2x.py', '../_base_/default_runtime.py'
]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
