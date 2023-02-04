_base_ = './pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d.py'

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

train_cfg = dict(max_epochs=24)
