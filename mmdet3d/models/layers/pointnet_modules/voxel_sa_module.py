# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import (stack_three_interpolate,
                      three_nn_vector_pool_by_two_step,
                      vector_pool_with_voxel_query)
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils.typing import InstanceList


class VectorPoolLocalInterpolateModule(nn.Module):
    """Vector pool local interpolate module.

    Args:
        mlp_channels (List[int]): MLP layer channels.
        num_voxels (List[int]): Number of grids in each local area.
            Include [num_grid_x, num_grid_y, num_grid_z].
        max_neighbour_distance (float): Max neighbour distance.
        nsample (int): Sample num in each neighbour area.
            If nsample==-1 find all, else find limited number(>0).
        neighbor_type (str): Query neighbor type. Include
            [ball, cube]. Default to 'cube'.
        use_xyz (bool): Whether used xyz coordinates as features.
            Default to True.
        neighbour_distance_multiplier (float): The multiplier of neighbor
            distance is used to calculate the distance by
            neighbor_distance_multiplier * max_neighbour_distance. Default
            to 1.0
        norm_cfg (dict, optional): Config dict of normalization layers. Default
            to dict(type='BN2d').
    """

    def __init__(
        self,
        mlp_channels: List[int],
        num_voxels: List[int],
        max_neighbour_distance: float,
        nsample: int,
        neighbor_type: str = 'cube',
        use_xyz: bool = True,
        neighbour_distance_multiplier: float = 1.0,
        norm_cfg: dict = dict(type='BN2d')
    ) -> None:
        super().__init__()
        self.num_voxels = num_voxels
        self.num_total_grids = self.num_voxels[0] * self.num_voxels[
            1] * self.num_voxels[2]
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_distance_multiplier = neighbour_distance_multiplier
        self.nsample = nsample
        assert neighbor_type in ['ball', 'cube']
        self.neighbor_type = neighbor_type
        self.use_xyz = use_xyz

        if mlp_channels is not None:
            if self.use_xyz:
                mlp_channels[0] += 9
            shared_mlps = []
            for k in range(len(mlp_channels) - 1):
                shared_mlps.extend(
                    ConvModule(
                        mlp_channels[k],
                        mlp_channels[k + 1],
                        kernel_size=1,
                        bias=False,
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                    ))
            self.mlp = nn.Sequential(*shared_mlps)
        else:
            self.mlp = None

        self.num_avg_length_of_neighbor_idxs = 1000

    def forward(self, support_xyz: Tensor, support_features: Tensor,
                batch_num_xyzs: Tensor, new_xyz: Tensor,
                new_xyz_grid_centers: Tensor,
                batch_num_new_xyzs: Tensor) -> Tensor:
        """
        Args:
            support_xyz (Tensor): Tensor of the xyz coordinates shape
                with (N1 + N2 ..., 3).
            support_features (Tensor): Features of each point with shape
                (N1 + N2 ..., C). C is features channel number.
            batch_num_xyzs: (Tensor): Stacked input xyz coordinates num in
                each batch, just like (N1, N2, ...).
            new_xyz (Tensor): Target points xyz coordinates shape with
                (M1 + M2 ..., 3).
            new_xyz_grid_centers (Tensor): Grids centers of each grid shape
                with  (M1 + M2 ..., num_total_grids, 3) .
            batch_num_new_xyzs: (Tensor): Stacked target points xyz coordinates
                num in each batch, just like (M1, M2, ...).

        Returns:
            Tensor: Target points features shape with
                (N1 + N2 ..., C_out).
        """
        with torch.no_grad():
            neighbor_type = 1 if self.neighbor_type == 'ball' else 0
            dist, idx, num_avg_length_of_neighbor_idxs = \
                three_nn_vector_pool_by_two_step(
                    support_xyz, batch_num_xyzs, new_xyz, new_xyz_grid_centers,
                    batch_num_new_xyzs, self.max_neighbour_distance,
                    self.nsample, neighbor_type,
                    self.num_avg_length_of_neighbor_idxs,
                    self.num_total_grids, self.neighbor_distance_multiplier)
        self.num_avg_length_of_neighbor_idxs = max(
            self.num_avg_length_of_neighbor_idxs,
            num_avg_length_of_neighbor_idxs.item())

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / torch.clamp_min(norm, min=1e-8)

        empty_mask = (idx.view(-1, 3)[:, 0] == -1)
        idx.view(-1, 3)[empty_mask] = 0

        interpolated_feats = stack_three_interpolate(support_features,
                                                     idx.view(-1, 3),
                                                     weight.view(-1, 3))
        interpolated_feats = interpolated_feats.view(
            idx.shape[0], idx.shape[1],
            -1)  # (M1 + M2 ..., num_total_grids, C)
        if self.use_xyz:
            near_known_xyz = support_xyz[idx.view(-1, 3).long()].view(
                -1, 3, 3)  # ( (M1 + M2 ...)*num_total_grids, 3)
            local_xyz = (new_xyz_grid_centers.view(-1, 1, 3) -
                         near_known_xyz).view(-1, idx.shape[1], 9)
            interpolated_feats = torch.cat(
                (interpolated_feats, local_xyz),
                dim=-1)  # ( M1 + M2 ..., num_total_grids, 9+C)

        new_features = interpolated_feats.view(
            -1, interpolated_feats.shape[-1]
        )  # ((M1 + M2 ...) * num_total_grids, C)
        new_features[empty_mask, :] = 0
        if self.mlp is not None:
            new_features = new_features.permute(
                1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
            new_features = self.mlp(new_features)

            new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(
                1, 0)  # (N1 + N2 ..., C)
        return new_features


class VectorPoolAggregationModule(nn.Module):
    """Vector pool aggregation module.

    Args:
        in_channels (int): Input channels.
        num_local_voxel (Tuple[int]): Number of grids in each local area.
            Include [num_grid_x, num_grid_y, num_grid_z]. Default to (3,3,3).
        local_aggregation_type (str): Type of local aggregation func. Include
            ['local_interpolation', 'voxel_avg_pool', 'voxel_random_choice'].
            Default to 'local_interpolation'.
        num_reduced_channels (int): Num reduced channels for vector pool
            module. Default to 1.
        num_aggregation_channels (int): Num of local aggregation model
            feature channels.
        post_mlps (Tuple[int]): Post encode mlp channels.
        neighbour_distance_multiplier (float): Multiplier of neighbor
            distance use to calculate distance by
            neighbor_distance_multiplier * max_neighbour_distance.
        max_neighbour_distance (float, optional): Max neighbour distance.
            Default to None.
        neighbor_nsample (int): Sample num in each neighbour area.
            If nsample==-1 find all, else find limited number(>0).
            Default to -1.
        neighbor_type (str): Query neighbor type. Include
            [ball, cube]. Default to 'cube'.
        use_xyz (bool): Whether use xyz coordinates as features.
            Default to True.
        norm_cfg (dict, optional): Config dict of normalization layers.
            Default to dict(type='BN2d').
    """

    def __init__(self,
                 in_channels: int,
                 num_local_voxel: Tuple[int] = (3, 3, 3),
                 local_aggregation_type: str = 'local_interpolation',
                 num_reduced_channels: int = 1,
                 num_aggregation_channels: int = 32,
                 post_mlps: Tuple[int] = (128, ),
                 neighbour_distance_multiplier: float = 2.0,
                 max_neighbour_distance: Optional[float] = None,
                 neighbor_nsample: int = -1,
                 neighbor_type: str = 'cube',
                 use_xyz: bool = True,
                 norm_cfg: dict = dict(type='BN2d')):
        super().__init__()
        self.num_local_voxel = num_local_voxel
        self.use_xyz = use_xyz
        self.total_voxels = \
            self.num_local_voxel[0] * self.num_local_voxel[1] \
            * self.num_local_voxel[2]
        self.local_aggregation_type = local_aggregation_type
        assert self.local_aggregation_type in [
            'local_interpolation', 'voxel_avg_pool', 'voxel_random_choice'
        ]
        self.input_channels = in_channels
        self.num_reduced_channels = in_channels \
            if num_reduced_channels is None else num_reduced_channels
        self.num_aggregation_channels = num_aggregation_channels
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_nsample = neighbor_nsample
        self.neighbor_type = neighbor_type

        if self.local_aggregation_type == 'local_interpolation':
            self.local_interpolate_module = \
                VectorPoolLocalInterpolateModule(
                    mlp=None,
                    num_voxels=self.num_local_voxel,
                    max_neighbour_distance=self.max_neighbour_distance,
                    nsample=self.neighbor_nsample,
                    neighbor_type=self.neighbor_type,
                    norm_cfg=norm_cfg,
                    neighbour_distance_multiplier=neighbour_distance_multiplier
                )
            num_c_in = (self.num_reduced_channels + 9) * self.total_voxels
        else:
            self.local_interpolate_module = None
            num_c_in = (self.num_reduced_channels + 3) * self.total_voxels

        num_c_out = self.total_voxels * self.num_aggregation_channels

        self.separate_local_aggregation_layer = nn.Sequential(
            nn.Conv1d(
                num_c_in,
                num_c_out,
                kernel_size=1,
                groups=self.total_voxels,
                bias=False), nn.BatchNorm1d(num_c_out), nn.ReLU()).cuda()

        post_mlp_list = []
        c_in = num_c_out
        for cur_num_c in post_mlps:
            post_mlp_list.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.post_mlps = nn.Sequential(*post_mlp_list).cuda()

        self.num_mean_points_per_grid = 20
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self) -> str:
        ret = f'radius={self.max_neighbour_distance},' \
              f' local_voxels=({self.num_local_voxel}, ' \
              f'local_aggregation_type={self.local_aggregation_type}, ' \
              f'num_c_reduction={self.input_channels}' \
              f'->{self.num_reduced_channels}, ' \
              f'num_c_local_aggregation=' \
              f'{self.num_aggregation_channels}'
        return ret

    @staticmethod
    def get_dense_voxels_by_center(point_centers: Tensor,
                                   max_neighbour_distance: float,
                                   num_voxels: List[int]) -> Tensor:
        """Get voxel centers in center points neighbour area.

        Args:
            point_centers (torch.Tensor): Center points coordinate,
                shape with (N, 3).
            max_neighbour_distance (float): Max neighbour distance.
            num_voxels (List[int]): Number of grids in each local area.

        Returns:
            torch.Tensor: Voxel centers, shape with (N, total_voxels, 3).
        """
        R = max_neighbour_distance
        device = point_centers.device
        x_grids = torch.arange(
            -R + R / num_voxels[0],
            R - R / num_voxels[0] + 1e-5,
            2 * R / num_voxels[0],
            device=device)
        y_grids = torch.arange(
            -R + R / num_voxels[1],
            R - R / num_voxels[1] + 1e-5,
            2 * R / num_voxels[1],
            device=device)
        z_grids = torch.arange(
            -R + R / num_voxels[2],
            R - R / num_voxels[2] + 1e-5,
            2 * R / num_voxels[2],
            device=device)
        x_offset, y_offset, z_offset = torch.meshgrid(
            x_grids, y_grids, z_grids)  # shape: [num_x, num_y, num_z]
        xyz_offset = torch.cat(
            (x_offset.contiguous().view(-1, 1), y_offset.contiguous().view(
                -1, 1), z_offset.contiguous().view(-1, 1)),
            dim=-1)
        voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
        return voxel_centers

    def vector_pool_with_local_interpolate(
            self, xyz: Tensor, batch_num_xyzs: Tensor, features: Tensor,
            new_xyz: Tensor, batch_num_new_xyzs: Tensor) -> Tensor:
        """
        Args:
            xyz (Tensor): Tensor of the xyz coordinates shape
                with (N1 + N2 ..., 3).
            batch_num_xyzs: (Tensor): Stacked input xyz coordinates num in
                each batch, just like (N1, N2, ...).
            features (Tensor): Features of each point shape with
                (N1 + N2 ..., C). C is feature channels number.
            new_xyz (Tensor): Target points xyz coordinates shape with
                (M1 + M2 ..., 3).
            batch_num_new_xyzs: (Tensor): Stacked target points xyz coordinates
                num in each batch, just like (M1, M2, ...).

        Returns:
            new_features: Target features after vector pool shape with
                (M, total_voxels * C).
        """
        voxel_centers = self.get_dense_voxels_by_center(
            point_centers=new_xyz,
            max_neighbour_distance=self.max_neighbour_distance,
            num_voxels=self.num_local_voxel
        )  # (M1 + M2 + ..., total_voxels, 3)
        voxel_features = self.local_interpolate_module.forward(
            support_xyz=xyz,
            support_features=features,
            batch_num_xyzs=batch_num_xyzs,
            new_xyz=new_xyz,
            new_xyz_grid_centers=voxel_centers,
            batch_num_new_xyzs=batch_num_new_xyzs
        )  # ((M1 + M2 ...) * total_voxels, C)

        voxel_features = voxel_features.contiguous().view(
            -1, self.total_voxels * voxel_features.shape[-1])
        return voxel_features

    def vector_pool_with_voxel_query(
            self, xyz: Tensor, batch_num_xyzs: Tensor, features: Tensor,
            new_xyz: Tensor, batch_num_new_xyzs: Tensor) -> Tuple[Tensor]:
        """
        Args:
            xyz (Tensor): Tensor of the xyz coordinates shape
                with (N1 + N2 ..., 3).
            batch_num_xyzs: (Tensor): Stacked input xyz coordinates num in
                each batch, just like (N1, N2, ...).
            features (Tensor): Features of each point with shape
                (N1 + N2 ..., C). C is features channel number.
            new_xyz (Tensor): Target points xyz coordinates shape with
                (M1 + M2 ..., 3).
            batch_num_new_xyzs: (Tensor): Stacked target points xyz coordinates
                num in each batch, just like (M1, M2, ...).

        Returns:
            new_features (Tensor): New features of target points
                neighbour area.
            point_cnt_of_grid (Tensor): Points num of each grid.
        """
        pooling_type = 0 \
            if self.local_aggregation_type == 'voxel_avg_pool' else 1
        use_xyz = int(self.use_xyz)
        new_features, new_local_xyz, num_mean_points_per_grid,\
            point_cnt_of_grid = vector_pool_with_voxel_query(
                xyz, batch_num_xyzs, features, new_xyz, batch_num_new_xyzs,
                self.num_local_voxel[0], self.num_local_voxel[1],
                self.num_local_voxel[2], self.max_neighbour_distance,
                self.num_reduced_channels, use_xyz,
                self.num_mean_points_per_grid, self.neighbor_nsample,
                self.neighbor_type, pooling_type)
        self.num_mean_points_per_grid = max(self.num_mean_points_per_grid,
                                            num_mean_points_per_grid.item())

        num_new_pts = new_features.shape[0]
        new_local_xyz = new_local_xyz.view(num_new_pts, -1,
                                           3)  # (N, num_voxel, 3)
        new_features = new_features.view(
            num_new_pts, -1, self.num_reduced_channels)  # (N, num_voxel, C)
        new_features = torch.cat((new_local_xyz, new_features),
                                 dim=-1).view(num_new_pts, -1)

        return new_features, point_cnt_of_grid

    def forward(self, xyz, batch_num_xyzs, new_xyz, batch_num_new_xyzs,
                features, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Args:
            xyz (Tensor): Tensor of the xyz coordinates shape
                with (N1 + N2 ..., 3).
            batch_num_xyzs: (Tensor): Stacked input xyz coordinates num in
                each batch, just like (N1, N2, ...).
            new_xyz (Tensor): Target points xyz coordinates shape with
                (M1 + M2 ..., 3).
            batch_num_new_xyzs: (Tensor): Stacked target points xyz coordinates
                num in each batch, just like (M1, M2, ...).
            features (Tensor): Features of each point shape with
                (N1 + N2 ..., C). C is feature channels num.
        """
        N, C = features.shape

        assert C % self.num_reduced_channels == 0, \
            f'the input channels ({C}) should be an integral multiple ' \
            f'of num_reduced_channels({self.num_reduced_channels})'

        features = features.view(N, -1, self.num_reduced_channels).sum(dim=1)

        if self.local_aggregation_type == 'local_interpolation':
            vector_features = self.vector_pool_with_local_interpolate(
                xyz=xyz,
                batch_num_xyzs=batch_num_xyzs,
                features=features,
                new_xyz=new_xyz,
                batch_num_new_xyzs=batch_num_new_xyzs
            )  # (M1 + M2 + ..., total_voxels * C)
        elif self.local_aggregation_type in [
                'voxel_avg_pool', 'voxel_random_choice'
        ]:
            vector_features, point_cnt_of_grid = \
                self.vector_pool_with_voxel_query(
                    xyz=xyz.contiguous(),
                    batch_num_xyzs=batch_num_xyzs,
                    features=features.contiguous(),
                    new_xyz=new_xyz.contiguous(),
                    batch_num_new_xyzs=batch_num_new_xyzs)
        else:
            raise NotImplementedError

        vector_features = vector_features.permute(
            1, 0)[None, :, :]  # (1, num_voxels * C, M1 + M2 ...)

        new_features = \
            self.separate_local_aggregation_layer(vector_features)

        new_features = self.post_mlps(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return new_xyz, new_features


@MODELS.register_module()
class VectorPoolAggregationModuleMSG(nn.Module):
    """Vector pool aggregation module with multi-scale grouping (MSG) used in
    PV RCNN++.

    Args:
        in_channels (int): Input channels.
        mlp_channels (List[List[int]]): Specify of the post encode mlp
            layer channels.
        local_aggregation_type (str): Type of local aggregation func.
            Include ['local_interpolation', 'voxel_avg_pool',
            'voxel_random_choice'].
        num_aggregation_channels (int): Num of local aggregation model
            feature channels.
        neighbour_distance_multiplier (float): The multiplier of neighbor
            distance is used to calculate the distance by
            neighbor_distance_multiplier * max_neighbour_distance.
        filter_neighbor_with_roi (bool): Whether use roi boxes to filter
            points. Default to True.
        roi_neighbour_radius (float): Sample points radius of each roi
            boxes. And needs filter_neighbour_with_roi = True. Default to 4.0
        part_max_points_num (int): Max points num in each part.
            Default to 200000.
        num_reduced_channels (int): Num reduced channels for vector pool
            module. Default to 1.
        groups_cfg_list (List[dict], optional): The config list of the vector
            pool module. Default to None.
    """

    def __init__(self,
                 in_channels: int,
                 mlp_channels: List[List[int]],
                 local_aggregation_type: str,
                 num_aggregation_channels: int,
                 neighbour_distance_multiplier: float = 2.0,
                 filter_neighbor_with_roi: bool = True,
                 roi_neighbour_radius: float = 4.0,
                 part_max_points_num: int = 200000,
                 num_reduced_channels: int = 1,
                 groups_cfg_list: Optional[List[dict]] = None,
                 **kwargs) -> None:
        super().__init__()
        self.filter_neighbor_with_roi = filter_neighbor_with_roi
        self.roi_neighbour_radius = roi_neighbour_radius
        self.part_max_points_num = part_max_points_num
        self.layers = nn.Sequential()
        c_in = 0
        for k, cur_config in enumerate(groups_cfg_list):
            cur_vector_pool_module = VectorPoolAggregationModule(
                in_channels=in_channels,
                num_local_voxel=cur_config.num_local_voxel,
                post_mlps=cur_config.post_mlps,
                max_neighbor_distance=cur_config.max_neighbor_distance,
                neighbor_nsample=cur_config.neighbor_nsample,
                local_aggregation_type=cur_config.get('local_aggregation_type',
                                                      local_aggregation_type),
                num_reduced_channels=cur_config.get('num_reduced_channels',
                                                    num_reduced_channels),
                num_aggregation_channels=cur_config.get(
                    'num_aggregation_channels', num_aggregation_channels),
                neighbour_distance_multiplier=cur_config.get(
                    'neighbour_distance_multiplier',
                    neighbour_distance_multiplier))
            self.layers.add_module(f'layer_{k}', cur_vector_pool_module)
            c_in += cur_config.post_mlps[-1]

        c_in += 3  # use_xyz

        shared_mlps = []
        for cur_num_c in mlp_channels:
            cur_num_c = cur_num_c[0]
            shared_mlps.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.msg_post_mlps = nn.Sequential(*shared_mlps)

    def sample_points_with_roi(self, rois: Tensor, points: Tensor) -> Tensor:
        """Sample points by roi boxes. Filter some points which keep away roi
        boxes.

        Args:
            rois (torch.Tensor): (M, 7 + C) Roi boxes.
            points (torch.Tensor): (N, 3) Input points cloud.

        Returns:
            torch.Tensor: (N_out, 3)  Sampled points close to roi boxes.
                N_out is sampled points num.
        """
        rois_center = rois[None, :, 0:3].clone()
        rois_center[:, :, 2] += rois[:, 5] / 2
        if points.shape[0] < self.part_max_points_num:

            distance = (points[:, None, :] - rois_center).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            point_mask = \
                min_dis < roi_max_dim + self.roi_neighbour_radius
        else:
            start_idx = 0
            point_mask_list = []
            while start_idx < points.shape[0]:
                distance = (points[start_idx:start_idx +
                                   self.part_max_points_num, None, :] -
                            rois_center).norm(dim=-1)
                min_dis, min_dis_roi_idx = distance.min(dim=-1)
                roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
                cur_point_mask = \
                    min_dis < roi_max_dim + self.roi_neighbour_radius
                point_mask_list.append(cur_point_mask)
                start_idx += self.part_max_points_num
            point_mask = torch.cat(point_mask_list, dim=0)

        sampled_points = points[:1] if point_mask.sum() == 0 else points[
            point_mask, :]
        return sampled_points, point_mask

    def forward(self,
                xyz: Tensor,
                batch_num_xyzs: Tensor,
                new_xyz: Tensor,
                batch_num_new_xyzs: Tensor,
                features: Optional[Tensor] = None,
                roi_boxes_list: Optional[InstanceList] = None,
                **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward.

        Args:
            xyz (Tensor): Tensor of the xyz coordinates shape
                with (N1 + N2 ..., 3).
            batch_num_xyzs: (Tensor): Stacked input xyz coordinates num in
                each batch, just like (N1, N2, ...).
            new_xyz (Tensor): Target points xyz coordinates shape with
                (M1 + M2 ..., 3).
            batch_num_new_xyzs: (Tensor): Stacked target points xyz coordinates
                num in each batch, just like (M1, M2, ...).
            features (Tensor, optional): Features of each point with shape
                (N1 + N2 ..., C). C is features channel number. Default: None.
            roi_boxes_list (List[:obj:`InstanceData`], optional):
                Roi boxes list. Default to None.

        Returns:
            Return target point coordinates and features:
                - cur_xyz  (Tensor): Target point coordinates with shape
                    (N1 + N2 ..., 3).
                - new_features (Tensor): Target point features with shape
                    (M1 + M2 ..., sum_k(mlps[k][-1])).
        """
        if roi_boxes_list is not None:
            if self.filter_neighbor_with_roi:
                point_features = torch.cat(
                    (xyz, features), dim=-1) if features is not None else xyz
                point_features_list = []
                cur_start = 0
                for batch_idx in range(len(batch_num_xyzs)):
                    _, valid_mask = self.sample_points_with_roi(
                        rois=roi_boxes_list[batch_idx],
                        points=xyz[cur_start:cur_start +
                                   batch_num_xyzs[batch_idx]])
                    point_features_list.append(
                        point_features[cur_start:cur_start +
                                       batch_num_xyzs[batch_idx]][valid_mask])
                    cur_start += batch_num_xyzs[batch_idx]
                    batch_num_xyzs[batch_idx] = valid_mask.sum()

                valid_point_features = torch.cat(point_features_list, dim=0)
                xyz = valid_point_features[:, 0:3]
                features = valid_point_features[:, 3:]\
                    if features is not None else None

        features_list = []
        for i in range(len(self.layers)):
            cur_xyz, cur_features = self.layers[i](xyz, batch_num_xyzs,
                                                   new_xyz, batch_num_new_xyzs,
                                                   features)
            features_list.append(cur_features)

        features = torch.cat(features_list, dim=-1)
        features = torch.cat((cur_xyz, features), dim=-1)
        features = features.permute(1, 0)[None, :, :]  # (1, C, N)
        new_features = self.msg_post_mlps(features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)  # (N, C)

        return cur_xyz, new_features
