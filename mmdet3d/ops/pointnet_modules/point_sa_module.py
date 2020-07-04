import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
from typing import List

from mmdet3d.ops import (GroupAll, QueryAndGroup, furthest_point_sample,
                         gather_points)


class PointSAModuleMSG(nn.Module):
    """Point set abstraction module with multi-scale grouping used in
    Pointnets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod='max',
                 normalize_xyz: bool = False):
        super().__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']

        self.num_point = num_point
        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

            mlp_spec = mlp_channels[i]
            if use_xyz:
                mlp_spec[0] += 3

            mlp = nn.Sequential()
            for i in range(len(mlp_spec) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg))
            self.mlps.append(mlp)

    def forward(
        self,
        points_xyz: torch.Tensor,
        features: torch.Tensor = None,
        indices: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is None:
            indices = furthest_point_sample(points_xyz, self.num_point)
        else:
            assert (indices.shape[1] == self.num_point)

        new_xyz = gather_points(xyz_flipped, indices).transpose(
            1, 2).contiguous() if self.num_point is not None else None

        for i in range(len(self.groupers)):
            # (B, C, num_point, nsample)
            new_features = self.groupers[i](points_xyz, new_xyz, features)

            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](new_features)
            if self.pool_mod == 'max':
                # (B, mlp[-1], num_point, 1)
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)])
            elif self.pool_mod == 'avg':
                # (B, mlp[-1], num_point, 1)
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], num_point)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices


class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module used in Pointnets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 num_point: int = None,
                 radius: float = None,
                 num_sample: int = None,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False):
        super().__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)
