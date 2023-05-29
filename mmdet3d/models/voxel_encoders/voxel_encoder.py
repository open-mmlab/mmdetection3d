# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from .utils import VFELayer, get_paddings_indicator


@MODELS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int, optional): Number of features to use. Default: 4.
    """

    def __init__(self, num_features: int = 4) -> None:
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
        """
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()


@MODELS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

    @torch.no_grad()
    def forward(self, features: Tensor, coors: Tensor, *args,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors


@MODELS.register_module()
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance of
            points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion
            layer used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: list = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_voxel_center: bool = False,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 fusion_layer: dict = None,
                 return_point_feats: bool = False):
        super(DynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    def forward(self,
                features: Tensor,
                coors: Tensor,
                points: Optional[Sequence[Tensor]] = None,
                img_feats: Optional[Sequence[Tensor]] = None,
                img_metas: Optional[dict] = None,
                *args,
                **kwargs) -> tuple:
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image features used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
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
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors


@MODELS.register_module()
class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance
            of points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance to
            center of voxel for each points inside a voxel. Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points inside a
            voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion layer
            used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the
            features of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: list = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_voxel_center: bool = False,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 fusion_layer: dict = None,
                 return_point_feats: bool = False):
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    def forward(self,
                features: Tensor,
                num_points: Tensor,
                coors: Tensor,
                img_feats: Optional[Sequence[Tensor]] = None,
                img_metas: Optional[dict] = None,
                *args,
                **kwargs) -> tuple:
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image features used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if (self.fusion_layer is not None and img_feats is not None):
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
                                                coors, img_feats, img_metas)

        return voxel_feats

    def fusion_with_mask(self, features: Tensor, mask: Tensor,
                         voxel_feats: Tensor, coors: Tensor,
                         img_feats: Sequence[Tensor],
                         img_metas: Sequence[dict]) -> Tensor:
        """Fuse image and point features with mask.

        Args:
            features (torch.Tensor): Features of voxel, usually it is the
                values of points in voxels.
            mask (torch.Tensor): Mask indicates valid features in each voxel.
            voxel_feats (torch.Tensor): Features of voxels.
            coors (torch.Tensor): Coordinates of each single voxel.
            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.
            img_metas (list(dict)): Meta information of image and points.

        Returns:
            torch.Tensor: Fused features of each voxel.
        """
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats,
                                        img_metas)

        voxel_canvas = voxel_feats.new_zeros(
            size=(voxel_feats.size(0), voxel_feats.size(1),
                  point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out


@MODELS.register_module()
class SegVFE(nn.Module):
    """Voxel feature encoder used in segmentation task.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 6.
        feat_channels (list(int)): Channels of features in VFE.
        with_voxel_center (bool): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float]): Size of a single voxel (rho, phi, z).
            Defaults to None.
        grid_shape (tuple[float]): The grid shape of voxelization.
            Defaults to (480, 360, 32).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Defaults to (0, -3.14159265359, -4, 50, 3.14159265359, 2).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        with_pre_norm (bool): Whether to use the norm layer before
            input vfe layer.
        feat_compression (int, optional): The voxel feature compression
            channels, Defaults to None
        return_point_feats (bool): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 6,
                 feat_channels: Sequence[int] = [],
                 with_voxel_center: bool = False,
                 voxel_size: Optional[Sequence[float]] = None,
                 grid_shape: Sequence[float] = (480, 360, 32),
                 point_cloud_range: Sequence[float] = (0, -3.14159265359, -4,
                                                       50, 3.14159265359, 2),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
                 mode: bool = 'max',
                 with_pre_norm: bool = True,
                 feat_compression: Optional[int] = None,
                 return_point_feats: bool = False) -> None:
        super(SegVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        assert not (voxel_size and grid_shape), \
            'voxel_size and grid_shape cannot be setting at the same time'
        if with_voxel_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        self.point_cloud_range = point_cloud_range
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

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = self.voxel_size[0]
        self.vy = self.voxel_size[1]
        self.vz = self.voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                vfe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                vfe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters), norm_layer,
                        nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.vfe_scatter = DynamicScatter(self.voxel_size,
                                          self.point_cloud_range,
                                          (mode != 'max'))
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression), nn.ReLU())

    def forward(self, features: Tensor, coors: Tensor, *args,
                **kwargs) -> Tuple[Tensor]:
        """Forward functions.

        Args:
            features (Tensor): Features of voxels, shape is NxC.
            coors (Tensor): Coordinates of voxels, shape is  Nx(1+NDim).

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels additionally.
        """
        features_ls = [features]

        # Find distance of x, y, and z from voxel center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 1].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 3].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        # Combine together feature decorations
        features = torch.cat(features_ls[::-1], dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = []
        for vfe in self.vfe_layers:
            features = vfe(features)
            point_feats.append(features)
        voxel_feats, voxel_coors = self.vfe_scatter(features, coors)

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        if self.return_point_feats:
            return voxel_feats, voxel_coors, point_feats
        return voxel_feats, voxel_coors
