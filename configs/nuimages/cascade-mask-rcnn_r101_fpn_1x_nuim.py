_base_ = './cascade_mask_rcnn_r50_fpn_1x_nuim.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
