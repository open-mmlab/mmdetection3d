_base_ = ['./minkunet_w32_8xb2-amp-3x_lpmix_semantickitti.py']

model = dict(
    data_preprocessor=dict(batch_first=False),
    backbone=dict(
        input_conv_module=dict(
            type='SparseConvModule', conv_cfg=dict(type='SubMConv3d')),
        downsample_module=dict(
            type='SparseConvModule', conv_cfg=dict(type='SparseConv3d')),
        upsample_module=dict(
            type='SparseConvModule',
            conv_cfg=dict(type='SparseInverseConv3d')),
        residual_block=dict(type='SparseBasicBlock'),
        sparseconv_backends='spconv'))
