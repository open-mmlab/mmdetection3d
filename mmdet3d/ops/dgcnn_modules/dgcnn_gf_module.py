# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from ..group_points import GroupAll, QueryAndGroup, grouping_operation


class BaseDGCNNGFModule(nn.Module):
    """Base module for point graph feature module used in DGCNN.

    Args:
        radii (list[float]): List of radius in each knn or ball query.
        sample_nums (list[int]): Number of samples in each knn or ball query.
        mlp_channels (list[list[int]]): Specify of the dgcnn before
            the global pooling for each graph feature module.
        knn_modes (list[str], optional): Type of KNN method, valid mode
            ['F-KNN', 'D-KNN'], Defaults to ['F-KNN'].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Defaults to False.
        use_xyz (bool, optional): Whether to use xyz as point features.
            Defaults to True.
        pool_mode (str, optional): Type of pooling method. Defaults to 'max'.
        normalize_xyz (bool, optional): If ball query, whether to normalize
            local XYZ with radius. Defaults to False.
        grouper_return_grouped_xyz (bool, optional): Whether to return grouped
            xyz in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool, optional): Whether to return grouped
            idx in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 radii,
                 sample_nums,
                 mlp_channels,
                 knn_modes=['F-KNN'],
                 dilated_group=False,
                 use_xyz=True,
                 pool_mode='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False):
        super(BaseDGCNNGFModule, self).__init__()

        assert len(sample_nums) == len(
            mlp_channels
        ), 'Num_samples and mlp_channels should have the same length.'
        assert pool_mode in ['max', 'avg'
                             ], "Pool_mode should be one of ['max', 'avg']."
        assert isinstance(knn_modes, list) or isinstance(
            knn_modes, tuple), 'The type of knn_modes should be list or tuple.'

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        self.pool_mode = pool_mode
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.knn_modes = knn_modes

        for i in range(len(sample_nums)):
            sample_num = sample_nums[i]
            if sample_num is not None:
                if self.knn_modes[i] == 'D-KNN':
                    grouper = QueryAndGroup(
                        radii[i],
                        sample_num,
                        use_xyz=use_xyz,
                        normalize_xyz=normalize_xyz,
                        return_grouped_xyz=grouper_return_grouped_xyz,
                        return_grouped_idx=True)
                else:
                    grouper = QueryAndGroup(
                        radii[i],
                        sample_num,
                        use_xyz=use_xyz,
                        normalize_xyz=normalize_xyz,
                        return_grouped_xyz=grouper_return_grouped_xyz,
                        return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mode == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mode == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(self, points):
        """forward.

        Args:
            points (Tensor): (B, N, C) input points.

        Returns:
            List[Tensor]: (B, N, C1) new points generated from each graph
                feature module.
        """
        new_points_list = [points]

        for i in range(len(self.groupers)):

            new_points = new_points_list[i]
            new_points_trans = new_points.transpose(
                1, 2).contiguous()  # (B, C, N)

            if self.knn_modes[i] == 'D-KNN':
                # (B, N, C) -> (B, N, K)
                idx = self.groupers[i](new_points[..., -3:].contiguous(),
                                       new_points[..., -3:].contiguous())[-1]

                grouped_results = grouping_operation(
                    new_points_trans, idx)  # (B, C, N) -> (B, C, N, K)
                grouped_results -= new_points_trans.unsqueeze(-1)
            else:
                grouped_results = self.groupers[i](
                    new_points, new_points)  # (B, N, C) -> (B, C, N, K)

            new_points = new_points_trans.unsqueeze(-1).repeat(
                1, 1, 1, grouped_results.shape[-1])
            new_points = torch.cat([grouped_results, new_points], dim=1)

            # (B, mlp[-1], N, K)
            new_points = self.mlps[i](new_points)

            # (B, mlp[-1], N)
            new_points = self._pool_features(new_points)
            new_points = new_points.transpose(1, 2).contiguous()
            new_points_list.append(new_points)

        return new_points


class DGCNNGFModule(BaseDGCNNGFModule):
    """Point graph feature module used in DGCNN.

    Args:
        mlp_channels (list[int]): Specify of the dgcnn before
            the global pooling for each graph feature module.
        num_sample (int, optional): Number of samples in each knn or ball
            query. Defaults to None.
        knn_mode (str, optional): Type of KNN method, valid mode
            ['F-KNN', 'D-KNN']. Defaults to 'F-KNN'.
        radius (float, optional): Radius to group with.
            Defaults to None.
        dilated_group (bool, optional): Whether to use dilated ball query.
            Defaults to False.
        norm_cfg (dict, optional): Type of normalization method.
            Defaults to dict(type='BN2d').
        act_cfg (dict, optional): Type of activation method.
            Defaults to dict(type='ReLU').
        use_xyz (bool, optional): Whether to use xyz as point features.
            Defaults to True.
        pool_mode (str, optional): Type of pooling method.
            Defaults to 'max'.
        normalize_xyz (bool, optional): If ball query, whether to normalize
            local XYZ with radius. Defaults to False.
        bias (bool | str, optional): If specified as `auto`, it will be decided
            by the norm_cfg. Bias will be set as True if `norm_cfg` is None,
            otherwise False. Defaults to 'auto'.
    """

    def __init__(self,
                 mlp_channels,
                 num_sample=None,
                 knn_mode='F-KNN',
                 radius=None,
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU'),
                 use_xyz=True,
                 pool_mode='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(DGCNNGFModule, self).__init__(
            mlp_channels=[mlp_channels],
            sample_nums=[num_sample],
            knn_modes=[knn_mode],
            radii=[radius],
            use_xyz=use_xyz,
            pool_mode=pool_mode,
            normalize_xyz=normalize_xyz,
            dilated_group=dilated_group)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]

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
                        act_cfg=act_cfg,
                        bias=bias))
            self.mlps.append(mlp)
