_base_ = [
    './minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_semantickitti.py'
]

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
