_base_ = './pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d.py'

model = dict(
    pts_backbone=dict(
        _delete_=True,
        type='NoStemRegNet',
        arch='regnetx_3.2gf',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf'),
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        strides=(1, 2, 2, 2),
        base_channels=64,
        stem_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(in_channels=[192, 432, 1008]))
