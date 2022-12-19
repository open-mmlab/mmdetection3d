# Copyright (c) OpenMMLab. All rights reserved.
from .dgcnn_head import DGCNNHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .segmentation_head import VoteSegHead

__all__ = ['PointNet2Head', 'DGCNNHead', 'PAConvHead','VoteSegHead']
