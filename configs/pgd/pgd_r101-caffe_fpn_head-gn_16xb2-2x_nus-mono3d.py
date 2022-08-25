_base_ = './pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
runner = dict(max_epochs=total_epochs)
