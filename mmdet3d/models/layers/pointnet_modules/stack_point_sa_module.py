# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import is_norm
from mmcv.ops import ball_query, grouping_operation
from torch import Tensor

from mmdet3d.registry import MODELS


class StackQueryAndGroup(nn.Module):
    """Find nearby points in spherical space.

    Args:
        radius (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        use_xyz (bool): Whether to use xyz. Default: True.
    """

    def __init__(self, radius: float, sample_nums: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.sample_nums, self.use_xyz = \
            radius, sample_nums, use_xyz

    def forward(self,
                xyz: torch.Tensor,
                xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor,
                new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None) -> Tuple[Tensor, Tensor]:
        """Forward.

        Args:
            xyz (Tensor): Tensor of the xyz coordinates
                of the features shape with (N1 + N2 ..., 3).
            xyz_batch_cnt: (Tensor): Stacked input xyz coordinates nums in
                each batch, just like (N1, N2, ...).
            new_xyz (Tensor): New coords of the outputs shape with
                (M1 + M2 ..., 3).
            new_xyz_batch_cnt: (Tensor): Stacked new xyz coordinates nums
                in each batch, just like (M1, M2, ...).
            features (Tensor, optional): Features of each point with shape
                (N1 + N2 ..., C). C is features channel number. Default: None.
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(
        ), 'xyz: %s, xyz_batch_cnt: %s' % (str(
            xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (
                str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(0, self.radius, self.sample_nums,
                                          xyz, new_xyz, xyz_batch_cnt,
                                          new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(
            xyz, idx, xyz_batch_cnt,
            new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0
        if features is not None:
            grouped_features = grouping_operation(
                features, idx, xyz_batch_cnt,
                new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features],
                    dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not' \
                                 ' use xyz as a feature!'
            new_features = grouped_xyz
        return new_features, idx


@MODELS.register_module()
class StackedSAModuleMSG(nn.Module):
    """Stack point set abstraction module.

    Args:
        in_channels (int): Input channels.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify mlp channels of the
            pointnet before the global pooling for each scale to encode
            point features.
        use_xyz (bool): Whether to use xyz. Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        norm_cfg (dict): Type of normalization method. Defaults to
            dict(type='BN2d', eps=1e-5, momentum=0.01).
    """

    def __init__(self,
                 in_channels: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 use_xyz: bool = True,
                 pool_mod='max',
                 norm_cfg: dict = dict(type='BN2d', eps=1e-5, momentum=0.01),
                 **kwargs) -> None:
        super().__init__()
        assert len(radii) == len(sample_nums) == len(mlp_channels)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            cin = in_channels
            if use_xyz:
                cin += 3
            radius = radii[i]
            nsample = sample_nums[i]
            mlp_spec = mlp_channels[i]

            self.groupers.append(
                StackQueryAndGroup(radius, nsample, use_xyz=use_xyz))

            mlp = nn.Sequential()
            for i in range(len(mlp_spec)):
                cout = mlp_spec[i]
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        cin,
                        cout,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=False))
                cin = cout
            self.mlps.append(mlp)
        self.pool_mod = pool_mod
        self.init_weights()

    def init_weights(self):
        """Initialize weights of StackedSAModuleMSG."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if is_norm(m):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                xyz: Tensor,
                xyz_batch_cnt: Tensor,
                new_xyz: Tensor,
                new_xyz_batch_cnt: Tensor,
                features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward.

        Args:
            xyz (Tensor): Tensor of the xyz coordinates
                of the features shape with (N1 + N2 ..., 3).
            xyz_batch_cnt: (Tensor): Stacked input xyz coordinates nums in
                each batch, just like (N1, N2, ...).
            new_xyz (Tensor): New coords of the outputs shape with
                (M1 + M2 ..., 3).
            new_xyz_batch_cnt: (Tensor): Stacked new xyz coordinates nums
                in each batch, just like (M1, M2, ...).
            features (Tensor, optional): Features of each point with shape
                (N1 + N2 ..., C). C is features channel number. Default: None.

        Returns:
            Return new points coordinates and features:
                - new_xyz  (Tensor): Target points coordinates with shape
                    (N1 + N2 ..., 3).
                - new_features (Tensor): Target points features with shape
                    (M1 + M2 ..., sum_k(mlps[k][-1])).
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            grouped_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features)  # (M1 + M2, Cin, nsample)
            grouped_features = grouped_features.permute(1, 0,
                                                        2).unsqueeze(dim=0)
            new_features = self.mlps[k](grouped_features)
            # (M1 + M2 ..., Cout, nsample)
            if self.pool_mod == 'max':
                new_features = new_features.max(-1).values
            elif self.pool_mod == 'avg':
                new_features = new_features.mean(-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)

        return new_xyz, new_features
