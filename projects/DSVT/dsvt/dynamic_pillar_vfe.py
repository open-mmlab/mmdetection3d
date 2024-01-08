# modified from https://github.com/Haiyang-W/DSVT
import numpy as np
import torch
import torch.nn as nn
import torch_scatter

from mmdet3d.registry import MODELS


class PFNLayerV2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


@MODELS.register_module()
class DynamicPillarVFE3D(nn.Module):
    """The difference between `DynamicPillarVFE3D` and `DynamicPillarVFE` is
    that the voxel in this module is along 3 dims: (x, y, z)."""

    def __init__(self, with_distance, use_absolute_xyz, use_norm, num_filters,
                 num_point_features, voxel_size, grid_size, point_cloud_range):
        super().__init__()
        self.use_norm = use_norm
        self.with_distance = with_distance
        self.use_absolute_xyz = use_absolute_xyz
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = num_filters
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(
                    in_filters,
                    out_filters,
                    self.use_norm,
                    last_layer=(i >= len(num_filters) - 2)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        point_cloud_range = np.array(point_cloud_range).astype(np.float32)
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        """Forward function.

        Args:
            batch_dict (dict[list]): Batch input data:
                - points [list[Tensor]]: list of batch input points.

        Returns:
            dict: Voxelization outputs:
                - points:
                - pillar_features/voxel_features:
                - voxel_coords
        """
        batch_prefix_points = []
        for batch_idx, points in enumerate(batch_dict['points']):
            prefix_batch_idx = torch.Tensor([batch_idx
                                             ]).tile(points.size(0),
                                                     1).to(points)
            prefix_points = torch.cat((prefix_batch_idx, points),
                                      dim=1)  # (batch_idx, x, y, z, i, e)
            batch_prefix_points.append(prefix_points)

        points = torch.cat(batch_prefix_points, dim=0)
        del prefix_points, batch_prefix_points

        points_coords = torch.floor(
            (points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) /
            self.voxel_size[[0, 1, 2]]).int()
        mask = ((points_coords >= 0) &
                (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
            points_coords[:, 0] * self.scale_yz + \
            points_coords[:, 1] * self.scale_z + points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(
            merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (
            points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x +
            self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (
            points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y +
            self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (
            points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z +
            self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack(
            (unq_coords // self.scale_xyz,
             (unq_coords % self.scale_xyz) // self.scale_yz,
             (unq_coords % self.scale_yz) // self.scale_z,
             unq_coords % self.scale_z),
            dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict
