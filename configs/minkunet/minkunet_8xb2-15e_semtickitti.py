_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/minkunet.py',
    '../_base_/schedules/cosine.py', '../_base_/default_runtime.py'
]

# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 2.4e-1  # max learning rate
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=lr, weight_decay=1.0e-4, momentum=0.9, nesterov=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=125),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=17940,
        by_epoch=False,
        eta_min=1e-5)
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

vis_backends = [dict(type='LocalVisBackend'),dict(type='WandbVisBackend', init_kwargs=dict(name='minku_8xb2-15e_sgd'))]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))