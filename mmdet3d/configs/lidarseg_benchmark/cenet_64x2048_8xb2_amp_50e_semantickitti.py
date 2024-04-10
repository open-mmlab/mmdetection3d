# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['./cenet_64x2048_8xb2_50e_semantickitti.py']

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
