_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_once.py',
    '../_base_/datasets/once-3d-5class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# optimizer, adapted from pcdet
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[35, 45])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=80)