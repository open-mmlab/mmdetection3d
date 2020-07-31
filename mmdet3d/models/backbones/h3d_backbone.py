import torch
from mmcv.runner import load_checkpoint
from torch import nn as nn
from torch.nn import functional as F

from mmdet.models import BACKBONES, build_backbone


@BACKBONES.register_module()
class H3DBackbone(nn.Module):
    """H3DBackbone with different config. '''Modified based on Ref:
    https://arxiv.org/abs/2006.05682 '''.

    Args:
        backbone (list): A list of backbone config.
    """

    def __init__(self, backbone=None, **kwarg):
        super().__init__()
        self.backbone_list = nn.ModuleList()
        self.suffix_list = []
        out_channel = 0
        for bb_cfg in backbone:
            self.suffix_list.append(bb_cfg['suffix'])
            bb_cfg.pop('suffix')
            out_channel += bb_cfg['fp_channels'][-1][-1]
            self.backbone_list.append(build_backbone(bb_cfg))

        # Feature concatenation
        self.conv_agg1 = torch.nn.Conv1d(out_channel, out_channel // 2, 1)
        self.bn_agg1 = torch.nn.BatchNorm1d(out_channel // 2)
        self.conv_agg2 = torch.nn.Conv1d(out_channel // 2, 256, 1)
        self.bn_agg2 = torch.nn.BatchNorm1d(256)

    def init_weights(self, pretrained=None):
        """Initialize the weights of PointNet++ backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of \
                    each fp features.
                - fp_features (list[torch.Tensor]): The features \
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the \
                    input points.
        """
        ret = {}
        fp_features = []
        for ind in range(len(self.backbone_list)):
            cur_ret = self.backbone_list[ind](points)
            cur_suffix = self.suffix_list[ind]
            fp_features.append(cur_ret['fp_features'][-1])
            if cur_suffix != '':
                for k in cur_ret.keys():
                    cur_ret[k + '_' + cur_suffix] = cur_ret.pop(k)
            ret.update(cur_ret)

        # Combine the feature here
        features_hd_discriptor = torch.cat(fp_features, dim=1)
        features_hd_discriptor = F.relu(
            self.bn_agg1(self.conv_agg1(features_hd_discriptor)))
        features_hd_discriptor = F.relu(
            self.bn_agg2(self.conv_agg2(features_hd_discriptor)))
        ret['hd_feature'] = features_hd_discriptor
        return ret
