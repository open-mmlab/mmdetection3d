_base_ = './fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
# model settings
model = dict(
    train_cfg=dict(
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05]))
# optimizer
optimizer = dict(lr=0.001)
load_from = 'work_dirs/fcos3d_nus/latest.pth'
