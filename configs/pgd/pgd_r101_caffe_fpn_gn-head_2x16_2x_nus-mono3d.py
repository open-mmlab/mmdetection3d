_base_ = './pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
runner = dict(max_epochs=total_epochs)
