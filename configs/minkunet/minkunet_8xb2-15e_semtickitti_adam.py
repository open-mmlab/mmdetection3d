_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/minkunet.py',
    '../_base_/schedules/cosine.py', '../_base_/default_runtime.py'
]

optim_wrapper = dict(clip_grad=dict(max_norm=35))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=15,
        by_epoch=True,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
