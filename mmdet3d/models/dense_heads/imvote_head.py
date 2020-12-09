from torch import nn as nn

from mmdet.models import HEADS


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
