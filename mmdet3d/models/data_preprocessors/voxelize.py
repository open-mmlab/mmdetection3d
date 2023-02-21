# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple, Union

import torch
from mmcv.utils import ext_loader
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

ext_module = ext_loader.load_ext('_ext', [
    'dynamic_voxelize_forward', 'hard_voxelize_forward',
    'dynamic_point_to_voxel_forward', 'dynamic_point_to_voxel_backward'
])


class _Voxelization(Function):

    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            voxel_size: Union[tuple, float],
            coors_range: Union[tuple, float],
            max_points: int = 35,
            max_voxels: int = 20000,
            deterministic: bool = True) -> Union[Tuple[torch.Tensor], Tuple]:
        """Convert kitti points(N, >=3) to voxels.

        Args:
            points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity.
            voxel_size (tuple or float): The size of voxel with the shape of
                [3].
            coors_range (tuple or float): The coordinate range of voxel with
                the shape of [6].
            max_points (int, optional): maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize. Default: 35.
            max_voxels (int, optional): maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
                Default: 20000.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
            elements. The first one is the output voxels with the shape of
            [M, max_points, n_dim], which only contain points and returned
            when max_points != -1. The second is the voxel coordinates with
            shape of [M, 3]. The last is number of point per voxel with the
            shape of [M], which only returned when max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            ext_module.dynamic_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                coors,
                NDim=3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = torch.zeros(size=(), dtype=torch.long)
            ext_module.hard_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                voxels,
                coors,
                num_points_per_voxel,
                voxel_num,
                max_points=max_points,
                max_voxels=max_voxels,
                NDim=3,
                deterministic=deterministic)
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class VoxelizationByGridShape(nn.Module):
    """Voxelization that allows inferring voxel size automatically based on
    grid shape.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        point_cloud_range (list):
            [x_min, y_min, z_min, x_max, y_max, z_max]
        max_num_points (int): max number of points per voxel
        voxel_size (list): list [x, y, z] or [rho, phi, z]
            size of single voxel.
        grid_shape (list): [L, W, H], grid shape of voxelization.
        max_voxels (tuple or int): max number of voxels in
            (training, testing) time
        deterministic: bool. whether to invoke the non-deterministic
            version of hard-voxelization implementations. non-deterministic
            version is considerablly fast but is not deterministic. only
            affects hard voxelization. default True. for more information
            of this argument and the implementation insights, please refer
            to the following links:
            https://github.com/open-mmlab/mmdetection3d/issues/894
            https://github.com/open-mmlab/mmdetection3d/pull/904
            it is an experimental feature and we will appreciate it if
            you could share with us the failing cases.
    """

    def __init__(self,
                 point_cloud_range: List,
                 max_num_points: int,
                 voxel_size: List = [],
                 grid_shape: List[int] = [],
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True):
        super().__init__()
        if voxel_size and grid_shape:
            raise ValueError('voxel_size is mutually exclusive grid_shape')
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] -
                          point_cloud_range[:3]) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (
                grid_shape - 1)
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels,
                            self.deterministic)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', grid_shape=' + str(self.grid_shape)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s


class _DynamicScatter(Function):
    """Different from the mmcv implementation, here it is allowed to return
    point2voxel_map."""

    @staticmethod
    def forward(ctx: Any,
                feats: torch.Tensor,
                coors: torch.Tensor,
                reduce_type: str = 'max',
                return_map: str = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats (torch.Tensor): [N, C]. Points features to be reduced
                into voxels.
            coors (torch.Tensor): [N, ndim]. Corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type (str, optional): Reduce op. support 'max', 'sum' and
                'mean'. Default: 'max'.
            return_map (str, optional): Whether to return point2voxel_map.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        results = ext_module.dynamic_point_to_voxel_forward(
            feats, coors, reduce_type)
        (voxel_feats, voxel_coors, point2voxel_map,
         voxel_points_count) = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map,
                              voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        if return_map:
            return voxel_feats, voxel_coors, point2voxel_map
        else:
            return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx: Any,
                 grad_voxel_feats: torch.Tensor,
                 grad_voxel_coors: Optional[torch.Tensor] = None) -> tuple:
        (feats, voxel_feats, point2voxel_map,
         voxel_points_count) = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        # TODO: whether to use index put or use cuda_backward
        # To use index put, need point to voxel index
        ext_module.dynamic_point_to_voxel_backward(
            grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats,
            point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter_3d = _DynamicScatter.apply


class DynamicScatter3D(nn.Module):
    """Scatters points into voxels, used in the voxel encoder with dynamic
    voxelization.

    Note:
        The CPU and GPU implementation get the same output, but have numerical
        difference after summation and division (e.g., 5e-7).

    Args:
        voxel_size (list): list [x, y, z] size of three dimension.
        point_cloud_range (list): The coordinate range of points, [x_min,
            y_min, z_min, x_max, y_max, z_max].
        average_points (bool): whether to use avg pooling to scatter points
            into voxel.
    """

    def __init__(self, voxel_size: List, point_cloud_range: List,
                 average_points: bool):
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(
            self, points: torch.Tensor,
            coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points into voxels.

        Args:
            points (torch.Tensor): Points to be reduced into voxels.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter_3d(points.contiguous(), coors.contiguous(),
                                  reduce)

    def forward(self, points: torch.Tensor,
                coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points/features into voxels.

        Args:
            points (torch.Tensor): Points to be reduced into voxels.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(
                    points[inds], coors[inds][:, 1:])
                coor_pad = F.pad(voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)

            return features, feature_coors

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', average_points=' + str(self.average_points)
        s += ')'
        return s
