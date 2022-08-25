_base_ = './pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d.py'
# model settings
model = dict(
    train_cfg=dict(code_weight=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2
    ]))
# optimizer
optimizer = dict(lr=0.002)
load_from = 'work_dirs/pgd_nus_benchmark_2x/latest.pth'
