# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.layers import PAConv
from mmdet3d.utils import ConfigType
from .builder import SA_MODULES


class BasePointSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (List[float]): List of radius in each ball query.
        sample_nums (List[int]): Number of samples in each ball query.
        mlp_channels (List[List[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (List[str]): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS']. Defaults to ['D-FPS'].

            - F-FPS: using feature distances for FPS.
            - D-FPS: using Euclidean distances of points for FPS.
            - FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (List[int]): Range of points to apply FPS.
            Defaults to [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Defaults to False.
        use_xyz (bool): Whether to use xyz. Defaults to True.
        pool_mod (str): Type of pooling method. Defaults to 'max'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Defaults to False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz
            in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx
            in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False,
                 grouper_return_grouped_xyz: bool = False,
                 grouper_return_grouped_idx: bool = False) -> None:
        super(BasePointSAModule, self).__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        if self.num_point is not None:
            self.points_sampler = Points_Sampler(self.num_point,
                                                 self.fps_mod_list,
                                                 self.fps_sample_range_list)
        else:
            self.points_sampler = None

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=grouper_return_grouped_xyz,
                    return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _sample_points(self, points_xyz: Tensor, features: Tensor,
                       indices: Tensor, target_xyz: Tensor) -> Tuple[Tensor]:
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) Features of each point.
            indices (Tensor): (B, num_point) Index of the features.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tuple[Tensor]:

            - new_xyz: (B, num_point, 3) Sampled xyz coordinates of points.
            - indices: (B, num_point) Sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            if self.num_point is not None:
                indices = self.points_sampler(points_xyz, features)
                new_xyz = gather_points(xyz_flipped,
                                        indices).transpose(1, 2).contiguous()
            else:
                new_xyz = None

        return new_xyz, indices

    def _pool_features(self, features: Tensor) -> Tensor:
        """Perform feature aggregation using pooling operation.

        Args:
            features (Tensor): (B, C, N, K) Features of locally grouped
                points before pooling.

        Returns:
            Tensor: (B, C, N) Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(
        self,
        points_xyz: Tensor,
        features: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        target_xyz: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """Forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) Features of each point.
                Defaults to None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Defaults to None.
            target_xyz (Tensor, optional): (B, M, 3) New coords of the outputs.
                Defaults to None.

        Returns:
            Tuple[Tensor]:

                - new_xyz: (B, M, 3) Where M is the number of points.
                  New features xyz.
                - new_features: (B, M, sum_k(mlps[k][-1])) Where M is the
                  number of points. New feature descriptors.
                - indices: (B, M) Where M is the number of points.
                  Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)

        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)

            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](grouped_results)

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices


@SA_MODULES.register_module()
class PointSAModuleMSG(BasePointSAModule):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (List[float]): List of radius in each ball query.
        sample_nums (List[int]): Number of samples in each ball query.
        mlp_channels (List[List[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (List[str]): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS']. Defaults to ['D-FPS'].

            - F-FPS: using feature distances for FPS.
            - D-FPS: using Euclidean distances of points for FPS.
            - FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (List[int]): Range of points to apply FPS.
            Defaults to [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN2d').
        use_xyz (bool): Whether to use xyz. Defaults to True.
        pool_mod (str): Type of pooling method. Defaults to 'max'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Defaults to False.
        bias (bool or str): If specified as `auto`, it will be decided by
            `norm_cfg`. `bias` will be set as True if `norm_cfg` is None,
            otherwise False. Defaults to 'auto'.
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 norm_cfg: ConfigType = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False,
                 bias: Union[bool, str] = 'auto') -> None:
        super(PointSAModuleMSG, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3

            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i],
                        mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)


@SA_MODULES.register_module()
class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (List[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int, optional): Number of points. Defaults to None.
        radius (float, optional): Radius to group with. Defaults to None.
        num_sample (int, optional): Number of samples in each ball query.
            Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Default to dict(type='BN2d').
        use_xyz (bool): Whether to use xyz. Defaults to True.
        pool_mod (str): Type of pooling method. Defaults to 'max'.
        fps_mod (List[str]): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS']. Defaults to ['D-FPS'].
        fps_sample_range_list (List[int]): Range of points to apply FPS.
            Defaults to [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Defaults to False.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 num_point: Optional[int] = None,
                 radius: Optional[float] = None,
                 num_sample: Optional[int] = None,
                 norm_cfg: ConfigType = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 normalize_xyz: bool = False) -> None:
        super(PointSAModule, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)
