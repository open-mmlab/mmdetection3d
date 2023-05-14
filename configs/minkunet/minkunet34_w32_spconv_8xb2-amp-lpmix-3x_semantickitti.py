_base_ = ['./minkunet34_w32_torchsparse_8xb2-lpmix-3x_semantickitti.py']

model = dict(
    data_preprocessor=dict(batch_first=True),
    backbone=dict(sparseconv_backends='spconv'))

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
