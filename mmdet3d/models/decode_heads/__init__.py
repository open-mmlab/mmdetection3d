# Copyright (c) OpenMMLab. All rights reserved.
from .dgcnn_head import DGCNNHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .mink_unet_head import MinkUNetHead

__all__ = ['PointNet2Head', 'DGCNNHead', 'PAConvHead']
