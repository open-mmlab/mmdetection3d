from torch import nn as nn

from mmdet.models import HEADS

# import numpy as np
# import torch
# from mmcv.runner import force_fp32
# from torch import nn as nn
# from torch.nn import functional as F

# from mmdet3d.core.post_processing import aligned_3d_nms
# from mmdet3d.models.builder import build_loss
# from mmdet3d.models.losses import chamfer_distance
# from mmdet3d.models.model_utils import VoteModule
# from mmdet3d.ops import build_sa_module, furthest_point_sample
# from mmdet.core import build_bbox_coder, multi_apply

# from .base_conv_bbox_head import BaseConvBboxHead


@HEADS.register_module()
class ImVoteHead(nn.Module):

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None):
        super(ImVoteHead,
              self).__init__(num_classes, bbox_coder, train_cfg, test_cfg,
                             vote_module_cfg, vote_aggregation_cfg,
                             pred_layer_cfg, conv_cfg, norm_cfg,
                             objectness_loss, center_loss, dir_class_loss,
                             dir_res_loss, size_class_loss, size_res_loss,
                             semantic_loss)
