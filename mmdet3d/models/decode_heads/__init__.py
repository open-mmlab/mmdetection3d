# Copyright (c) OpenMMLab. All rights reserved.
from .cylinder3d_head import Cylinder3DHead
from .dgcnn_head import DGCNNHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head

__all__ = ['PointNet2Head', 'DGCNNHead', 'PAConvHead', 'Cylinder3DHead']
