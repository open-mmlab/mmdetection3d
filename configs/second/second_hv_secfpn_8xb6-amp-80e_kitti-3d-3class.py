_base_ = 'second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'

# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)
