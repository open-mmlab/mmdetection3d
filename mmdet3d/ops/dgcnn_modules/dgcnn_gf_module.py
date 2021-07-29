import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from ..group_points import GroupAll, QueryAndGroup, grouping_operation
from ..paconv import PAConv
from .builder import GF_MODULES


class BaseDGCNNGFModule(nn.Module):
    """Base module for point graph feature module used in DGCNN.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each knn or ball query.
        sample_nums (list[int]): Number of samples in each knn or ball query.
        mlp_channels (list[list[int]]): Specify of the dgcnn before
            the global pooling for each graph feature module.
        knn_mod (list[str]: Type of KNN method, valid mod
            ['F-KNN', 'D-KNN'], Default: ['F-KNN'].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz in
            `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx in
            `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 knn_mod=['F-KNN'],
                 dilated_group=False,
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False):
        super(BaseDGCNNGFModule, self).__init__()

        assert len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(knn_mod, list) or isinstance(knn_mod, tuple)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.knn_mod = knn_mod

        for i in range(len(sample_nums)):
            sample_num = sample_nums[i]
            if num_point is not None:
                if self.knn_mod[i] == 'D-KNN':
                    grouper = QueryAndGroup(
                        radii[i],
                        sample_num,
                        use_xyz=use_xyz,  # True
                        normalize_xyz=normalize_xyz,  # False
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

            if self.knn_mod[i] == 'D-KNN':
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

            # (B, mlp[-1], num_point, nsample)
            new_points = self.mlps[i](new_points)

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_points, tuple)
                new_points = new_points[0]

            # (B, mlp[-1], num_point)
            new_points = self._pool_features(new_points)
            new_points = new_points.transpose(1, 2).contiguous()
            new_points_list.append(new_points)

        return new_points


@GF_MODULES.register_module()
class DGCNNGFModule(BaseDGCNNGFModule):
    """Point graph feature module used in DGCNN.

    Args:
        mlp_channels (list[int]): Specify of the dgcnn before
            the global pooling for each graph feature module.
        num_point (int): Number of points.
            Default: None.
        num_sample (int): Number of samples in each knn or ball query.
            Default: None.
        knn_mod (list[str]: Type of KNN method, valid mod
            ['F-KNN', 'D-KNN'], Default: ['F-KNN'].
        radius (float): Radius to group with.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        act_cfg (dict): Type of activation method.
            Default: dict(type='ReLU').
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 num_sample=None,
                 knn_mod=['F-KNN'],
                 radius=None,
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(DGCNNGFModule, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            sample_nums=[num_sample],
            knn_mod=[knn_mod],
            radii=[radius],
            use_xyz=use_xyz,
            pool_mod=pool_mod,
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
