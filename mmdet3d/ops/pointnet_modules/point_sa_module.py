import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
from typing import List, Tuple

from mmdet3d.ops import (GroupAll, PAConv, PAConvCUDA, Points_Sampler,
                         QueryAndGroup, gather_points)
from .builder import SA_MODULES


class _PointSAModuleBase(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz in
            `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx in
            `QueryAndGroup`. Defaults to False.
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
                 grouper_return_grouped_idx: bool = False):
        super(_PointSAModuleBase, self).__init__()

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
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        self.points_sampler = Points_Sampler(self.num_point, self.fps_mod_list,
                                             self.fps_sample_range_list)

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

    def _sample_points(
            self, points_xyz: torch.Tensor, features: torch.Tensor,
            indices: torch.Tensor,
            target_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, num_point, 3) sampled xyz coordinates of points.
            Tensor: (B, num_point) sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            indices = self.points_sampler(points_xyz, features)
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None

        return new_xyz, indices

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)

        Returns:
            torch.Tensor: (B, C, N)
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
        points_xyz: torch.Tensor,
        features: torch.Tensor = None,
        indices: torch.Tensor = None,
        target_xyz: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
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
class PointSAModuleMSG(_PointSAModuleBase):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False,
                 bias: str = 'auto'):
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
            mlp_spec = self.mlp_channels[i]
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
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)


@SA_MODULES.register_module()
class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

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
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
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
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 normalize_xyz: bool = False):
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


@SA_MODULES.register_module()
class PAConvSAModuleMSG(_PointSAModuleBase):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PAConv networks.

    Replace the MLPs in `PointSAModuleMSG` with PAConv layers.
    See the `paper <https://arxiv.org/abs/2103.14635>`_ for more detials.

    Args:
        paconv_num_kernels (list[list[int]]): Number of weight kernels in the
            weight banks of each layer's PAConv.
        paconv_kernel_input (str, optional): Input features to be multiplied
            with weight kernels. Can be 'identity' or 'neighbor'.
            Defaults to 'neighbor'.
        scorenet_input (str, optional): Type of the input to ScoreNet.
            Can be 'identity', 'neighbor' or 'ed7'. Defaults to 'ed7'.
        scorenet_cfg (dict, optional): Config of the ScoreNet module, which
            may contain the following keys and values:

            - mlp_channels (List[int]): Hidden units of MLPs.
            - score_norm (str): Normalization function of output scores.
                Can be 'softmax', 'sigmoid' or 'identity'.
            - temp_factor (float): Temperature factor to scale the output
                scores before softmax.
            - last_bn (bool): Whether to use BN on the last output of mlps.
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 paconv_num_kernels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 norm_cfg: dict = dict(type='BN2d', momentum=0.1),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False,
                 bias: str = 'auto',
                 paconv_kernel_input: str = 'neighbor',
                 scorenet_input: str = 'ed7',
                 scorenet_cfg: dict = dict(
                     mlp_channels=[16, 16, 16],
                     score_norm='softmax',
                     temp_factor=1.0,
                     last_bn=False)):
        super(PAConvSAModuleMSG, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz,
            grouper_return_grouped_xyz=True)

        assert len(paconv_num_kernels) == len(mlp_channels)
        for i in range(len(mlp_channels)):
            assert len(paconv_num_kernels[i]) == len(mlp_channels[i]) - 1, \
                'PAConv number of weight kernels wrong'

        # in PAConv, bias only exists in ScoreNet
        scorenet_cfg['bias'] = bias

        for i in range(len(self.mlp_channels)):
            mlp_spec = self.mlp_channels[i]
            if use_xyz:
                mlp_spec[0] += 3

            num_kernels = paconv_num_kernels[i]

            mlp = nn.Sequential()
            for i in range(len(mlp_spec) - 1):
                mlp.add_module(
                    f'layer{i}',
                    PAConv(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        num_kernels[i],
                        norm_cfg=norm_cfg,
                        kernel_input=paconv_kernel_input,
                        scorenet_input=scorenet_input,
                        scorenet_cfg=scorenet_cfg))
            self.mlps.append(mlp)


@SA_MODULES.register_module()
class PAConvSAModule(PAConvSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PAConv networks.

    Replace the MLPs in `PointSAModule` with PAConv layers. See the `paper
    <https://arxiv.org/abs/2103.14635>`_ for more detials.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 paconv_num_kernels: List[int],
                 num_point: int = None,
                 radius: float = None,
                 num_sample: int = None,
                 norm_cfg: dict = dict(type='BN2d', momentum=0.1),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 normalize_xyz: bool = False,
                 paconv_kernel_input: str = 'neighbor',
                 scorenet_input: str = 'ed7',
                 scorenet_cfg: dict = dict(
                     mlp_channels=[16, 16, 16],
                     score_norm='softmax',
                     temp_factor=1.0,
                     last_bn=False)):
        super(PAConvSAModule, self).__init__(
            mlp_channels=[mlp_channels],
            paconv_num_kernels=[paconv_num_kernels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz,
            paconv_kernel_input=paconv_kernel_input,
            scorenet_input=scorenet_input,
            scorenet_cfg=scorenet_cfg)


@SA_MODULES.register_module()
class PAConvSAModuleMSGCUDA(_PointSAModuleBase):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PAConv networks.

    Replace the non CUDA version PAConv with CUDA implemented PAConv for
    efficient computation. See the `paper <https://arxiv.org/abs/2103.14635>`_
    for more detials.
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 paconv_num_kernels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 norm_cfg: dict = dict(type='BN2d', momentum=0.1),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False,
                 bias: str = 'auto',
                 paconv_kernel_input: str = 'neighbor',
                 scorenet_input: str = 'ed7',
                 scorenet_cfg: dict = dict(
                     mlp_channels=[8, 16, 16],
                     score_norm='softmax',
                     temp_factor=1.0,
                     last_bn=False)):
        super(PAConvSAModuleMSGCUDA, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz,
            grouper_return_grouped_xyz=True,
            grouper_return_grouped_idx=True)

        assert len(paconv_num_kernels) == len(mlp_channels)
        for i in range(len(mlp_channels)):
            assert len(paconv_num_kernels[i]) == len(mlp_channels[i]) - 1, \
                'PAConv number of weight kernels wrong'

        # in PAConv, bias only exists in ScoreNet
        scorenet_cfg['bias'] = bias

        # we need to manually concat xyz for CUDA implemented PAConv
        self.use_xyz = use_xyz

        for i in range(len(self.mlp_channels)):
            mlp_spec = self.mlp_channels[i]
            if use_xyz:
                mlp_spec[0] += 3

            num_kernels = paconv_num_kernels[i]

            # can't use `nn.Sequential` for PAConvCUDA because its input and
            # output have different shapes
            mlp = nn.ModuleList()
            for i in range(len(mlp_spec) - 1):
                mlp.append(
                    PAConvCUDA(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        num_kernels[i],
                        norm_cfg=norm_cfg,
                        kernel_input=paconv_kernel_input,
                        scorenet_input=scorenet_input,
                        scorenet_cfg=scorenet_cfg))
            self.mlps.append(mlp)

    def forward(
        self,
        points_xyz: torch.Tensor,
        features: torch.Tensor = None,
        indices: torch.Tensor = None,
        target_xyz: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)

        for i in range(len(self.groupers)):
            xyz = points_xyz
            new_features = features
            for j in range(len(self.mlps[i])):
                # we don't use grouped_features here to avoid large GPU memory
                # _, (B, 3, num_point, nsample), (B, num_point, nsample)
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz,
                                                               new_features)

                # concat xyz as additional features
                if self.use_xyz and j == 0:
                    # (B, C+3, N)
                    new_features = torch.cat(
                        (points_xyz.permute(0, 2, 1), new_features), dim=1)

                # (B, out_c, num_point, nsample)
                grouped_new_features = self.mlps[i][j](
                    (new_features, grouped_xyz, grouped_idx.long()))[0]

                # different from PointNet++ and non CUDA version of PAConv
                # CUDA version of PAConv needs to aggregate local features
                # every time after it passes through a Conv layer
                # in order to transform to valid input shape
                # (B, out_c, num_point)
                new_features = self._pool_features(grouped_new_features)

                # constrain the points to be grouped for next PAConv layer
                # because new_features only contains sampled centers now
                # (B, num_point, 3)
                xyz = new_xyz

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices


@SA_MODULES.register_module()
class PAConvSAModuleCUDA(PAConvSAModuleMSGCUDA):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PAConv networks.

    Replace the non CUDA version PAConv with CUDA implemented PAConv for
    efficient computation. See the `paper <https://arxiv.org/abs/2103.14635>`_
    for more detials.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 paconv_num_kernels: List[int],
                 num_point: int = None,
                 radius: float = None,
                 num_sample: int = None,
                 norm_cfg: dict = dict(type='BN2d', momentum=0.1),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 normalize_xyz: bool = False,
                 paconv_kernel_input: str = 'neighbor',
                 scorenet_input: str = 'ed7',
                 scorenet_cfg: dict = dict(
                     mlp_channels=[8, 16, 16],
                     score_norm='softmax',
                     temp_factor=1.0,
                     last_bn=False)):
        super(PAConvSAModuleCUDA, self).__init__(
            mlp_channels=[mlp_channels],
            paconv_num_kernels=[paconv_num_kernels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz,
            paconv_kernel_input=paconv_kernel_input,
            scorenet_input=scorenet_input,
            scorenet_cfg=scorenet_cfg)
