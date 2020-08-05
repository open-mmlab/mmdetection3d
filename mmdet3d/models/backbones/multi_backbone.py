import copy
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from torch import nn as nn

from mmdet.models import BACKBONES, build_backbone


@BACKBONES.register_module()
class MultiBackbone(nn.Module):
    """MultiBackbone with different config. '''.

    Args:
        num_stream (int): The number of backbones.
        backbones (list or dict): A list of backbone configs.
        aggregation_mlp_channels (list[int]): Specify the mlp layers
            for feature aggregation.
        conv_cfg (dict): Config dict of convolutional layers
        norm_cfg (dict): Config dict of normalization layers
        act_cfg (dict): Config dict of activation layers
    """

    def __init__(self,
                 num_stream,
                 backbones,
                 aggregation_mlp_channels=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        assert isinstance(backbones, dict) or isinstance(backbones, list)
        if isinstance(backbones, dict):
            backbones_list = []
            for ind in range(num_stream):
                backbones_list.append(copy.deepcopy(backbones))
                backbones_list[-1]['suffix'] = 'net{}'.format(ind)
            backbones = backbones_list

        assert len(backbones) == num_stream

        self.backbone_list = nn.ModuleList()
        # Rename the ret_dict with different suffixs.
        self.suffix_list = []

        out_channels = 0

        for bb_cfg in backbones:
            assert 'suffix' in bb_cfg.keys()
            # assert 'collector' in bb_cfg.keys()

            self.suffix_list.append(bb_cfg['suffix'])
            bb_cfg.pop('suffix')

            out_channels += bb_cfg['fp_channels'][-1][-1]
            self.backbone_list.append(build_backbone(bb_cfg))

        # Feature aggregation layers
        if aggregation_mlp_channels is None:
            aggregation_mlp_channels = [
                out_channels, out_channels // 2,
                out_channels // len(self.backbone_list)
            ]
        else:
            aggregation_mlp_channels.insert(0, out_channels)

        self.aggregation_layers = nn.Sequential()
        for i in range(len(aggregation_mlp_channels) - 1):
            self.aggregation_layers.add_module(
                f'layer{i}',
                ConvModule(
                    aggregation_mlp_channels[i],
                    aggregation_mlp_channels[i + 1],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=True))

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
                - hd_feature (torch.Tensor): The aggregation feature \
                    from multiple backbones.
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
        features_hd_discriptor = self.aggregation_layers(
            features_hd_discriptor)
        ret['hd_feature'] = features_hd_discriptor
        return ret
