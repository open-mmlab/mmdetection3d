import torch
from mmcv.cnn import build_conv_layer
from mmengine import init_default_scope
from mmengine.runner import load_checkpoint

init_default_scope('mmdet3d')

dcn_cfg = dict(type='DCNv2', deformable_groups=1, bias=False)
dcn = build_conv_layer(dcn_cfg, 256, 256, 3).cuda()
load_checkpoint(dcn, '../TPVFormer/dcn.pth')

x = torch.load('../TPVFormer/x.pth')
y = dcn(x)
pass
