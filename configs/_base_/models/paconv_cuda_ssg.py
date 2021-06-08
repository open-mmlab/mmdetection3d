_base_ = './paconv_ssg.py'

model = dict(
    backbone=dict(
        sa_cfg=dict(
            type='PAConvCUDASAModule',
            scorenet_cfg=dict(mlp_channels=[8, 16, 16]))))
