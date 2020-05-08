import torch
from mmcv.cnn import build_norm_layer
from torch import nn

from mmdet3d.ops import DynamicScatter
from ..registry import VOXEL_ENCODERS
from .utils import PFNLayer, get_paddings_indicator


@VOXEL_ENCODERS.register_module()
class PillarFeatureNet(nn.Module):

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max'):
        """ Pillar Feature Net.
        The network prepares the pillar features and performs forward pass
        through PFNLayers.

        Args:
            num_input_features (int). Number of input features,
                either x, y, z or x, y, z, r.
            use_norm (bool). Whether to include BatchNorm.
            num_filters (list[int]). Number of features in each of the
                N PFNLayers.
            with_distance (bool). Whether to include Euclidean distance
                to points.
            voxel_size (list[float]). Size of voxels, only utilize x and y
                size.
            point_cloud_range (list[float>]). Point cloud range, only
                utilize x and y min.
        """

        super(PillarFeatureNet, self).__init__()
        assert len(num_filters) > 0
        if with_cluster_center:
            num_input_features += 3
        if with_voxel_center:
            num_input_features += 2
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        # Create PillarFeatureNet layers
        self.num_input_features = num_input_features
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    use_norm,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features[:, :, :2]
            f_center[:, :, 0] = f_center[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()


@VOXEL_ENCODERS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max'):
        """
        Dynamic Pillar Feature Net for Dynamic Voxelization.
        The difference is in the forward part
        """

        super(DynamicPillarFeatureNet, self).__init__(
            num_input_features,
            use_norm,
            num_filters,
            with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            mode=mode)

        num_filters = [self.num_input_features] + list(num_filters)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def forward(self, features, coors):
        """
        features (torch.Tensor): NxC
        coors (torch.Tensor): Nx(1+NDim)
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 2))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class AlignedPillarFeatureNet(nn.Module):

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max'):
        """ Pillar Feature Net.

        The network prepares the pillar features and performs forward pass
        through PFNLayers.

        Args:
            num_input_features (int): Number of input features, either x, y, z
                or x, y, z, r.
            use_norm (bool): Whether to include BatchNorm.
            num_filters (list[int]): Number of features in each of the N
                PFNLayers.
            with_distance (bool): Whether to include Euclidean distance to
                points.
            voxel_size (list[float]): Size of voxels, only utilize x and y
                size.
            point_cloud_range: (list[float]): Point cloud range, only
                utilize x and y min.
        """

        super(AlignedPillarFeatureNet, self).__init__()

        assert len(num_filters) > 0
        if with_cluster_center:
            print('Use cluster center')
            num_input_features += 3
        if with_voxel_center:
            print('Use voxel center')
            num_input_features += 2
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    use_norm,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to
        # calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

    def forward(self, features, num_points, coors):
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        x_distance = features[:, :, 0] - (
            coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
            self.x_offset)
        y_distance = features[:, :, 1] - (
            coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
            self.y_offset)
        z_distance = features[:, :, 2] - (
            coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
            self.z_offset)

        normed_x_distance = 1 - torch.abs(x_distance / self.vx)
        normed_y_distance = 1 - torch.abs(y_distance / self.vy)
        normed_z_distance = 1 - torch.abs(z_distance / self.vz)

        x_mask = torch.gt(normed_x_distance, 0).type_as(features)
        y_mask = torch.gt(normed_y_distance, 0).type_as(features)
        z_mask = torch.gt(normed_z_distance, 0).type_as(features)

        nonzero_points_mask = x_mask.mul(y_mask).mul(z_mask)
        aligned_distance = normed_x_distance.mul(normed_y_distance).mul(
            normed_z_distance).mul(nonzero_points_mask)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features[:, :, :2]
            f_center[:, :, 0] = f_center[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to
        # whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            if pfn.last_vfe:
                features = pfn(features, aligned_distance)
            else:
                features = pfn(features)

        return features.squeeze()
