# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import build_sa_module
from ..builder import BACKBONES
from .base_pointnet import BasePointNet


@BACKBONES.register_module()
class PointNet2SAMSG(BasePointNet):
    """PointNet2 with Multi-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods (tuple[int]): Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
                 num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
                 sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                              ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                              ((128, 128, 256), (128, 192, 256), (128, 256,
                                                                  256))),
                 aggregation_channels=(64, 128, 256),
                 fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
                 fps_sample_range_lists=((-1), (-1), (512, -1)),
                 dilated_group=(True, True, True),
                 out_indices=(2, ),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModuleMSG',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=False),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.out_indices = out_indices
        assert max(out_indices) < self.num_sa
        assert len(num_points) == len(radii) == len(num_samples) == len(
            sa_channels)
        if aggregation_channels is not None:
            assert len(sa_channels) == len(aggregation_channels)
        else:
            aggregation_channels = [None] * len(sa_channels)

        self.SA_modules = nn.ModuleList()
        self.aggregation_mlps = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(radii[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radii=radii[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    dilated_group=dilated_group[sa_index],
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg,
                    bias=True))
            skip_channel_list.append(sa_out_channel)

            cur_aggregation_channel = aggregation_channels[sa_index]
            if cur_aggregation_channel is None:
                self.aggregation_mlps.append(None)
                sa_in_channel = sa_out_channel
            else:
                self.aggregation_mlps.append(
                    ConvModule(
                        sa_out_channel,
                        cur_aggregation_channel,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d'),
                        kernel_size=1,
                        bias=True))
                sa_in_channel = cur_aggregation_channel

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        out_sa_xyz = [xyz]
        out_sa_features = [features]
        out_sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            if self.aggregation_mlps[i] is not None:
                cur_features = self.aggregation_mlps[i](cur_features)
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])
                out_sa_indices.append(sa_indices[-1])

        return dict(
            sa_xyz=out_sa_xyz,
            sa_features=out_sa_features,
            sa_indices=out_sa_indices)
