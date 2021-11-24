_base_ = '../second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
