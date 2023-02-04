# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import (get_proj_mat_by_coord_type,
                                        points_cam2img, points_img2cam)
from mmdet3d.utils import OptConfigType, OptMultiConfig
from . import apply_3d_transformation


def point_sample(img_meta: dict,
                 img_features: Tensor,
                 points: Tensor,
                 proj_mat: Tensor,
                 coord_type: str,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True,
                 valid_flag: bool = False) -> Tensor:
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): 1 x C x H x W image features.
        points (Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.
        valid_flag (bool): Whether to filter out the points that outside
            the image and with depth smaller than 0. Defaults to False.

    Returns:
        Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to image coordinate
    if valid_flag:
        proj_pts = points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (
                depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()


@MODELS.register_module()
class PointFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (List[int] or int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (List[int] or int): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'. Defaults to 'LIDAR'.
        conv_cfg (:obj:`ConfigDict` or dict): Config dict for convolution
            layers of middle layers. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layers of middle layers. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
        activate_out (bool): Whether to apply relu activation to output
            features. Defaults to True.
        fuse_out (bool): Whether to apply conv layer to the fused features.
            Defaults to False.
        dropout_ratio (int or float): Dropout ratio of image features to
            prevent overfitting. Defaults to 0.
        aligned (bool): Whether to apply aligned feature fusion.
            Defaults to True.
        align_corners (bool): Whether to align corner when sampling features
            according to points. Defaults to True.
        padding_mode (str): Mode used to pad the features of points that do not
            have corresponding image features. Defaults to 'zeros'.
        lateral_conv (bool): Whether to apply lateral convs to image features.
            Defaults to True.
    """

    def __init__(self,
                 img_channels: Union[List[int], int],
                 pts_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 img_levels: Union[List[int], int] = 3,
                 coord_type: str = 'LIDAR',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 activate_out: bool = True,
                 fuse_out: bool = False,
                 dropout_ratio: Union[int, float] = 0,
                 aligned: bool = True,
                 align_corners: bool = True,
                 padding_mode: str = 'zeros',
                 lateral_conv: bool = True) -> None:
        super(PointFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats: List[Tensor], pts: List[Tensor],
                pts_feats: Tensor, img_metas: List[dict]) -> Tensor:
        """Forward function.

        Args:
            img_feats (List[Tensor]): Image features.
            pts: (List[Tensor]): A batch of points with shape N x 3.
            pts_feats (Tensor): A tensor consist of point features of the
                total batch.
            img_metas (List[dict]): Meta information of images.

        Returns:
            Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats: List[Tensor], pts: List[Tensor],
                          img_metas: List[dict]) -> Tensor:
        """Obtain multi-level features for each point.

        Args:
            img_feats (List[Tensor]): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (List[Tensor]): Points of each sample.
            img_metas (List[dict]): Meta information for each sample.

        Returns:
            Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats: Tensor, pts: Tensor,
                      img_meta: dict) -> Tensor:
        """Sample features from single level image feature map.

        Args:
            img_feats (Tensor): Image feature map in shape (1, C, H, W).
            pts (Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts


def voxel_sample(voxel_features: Tensor,
                 voxel_range: List[float],
                 voxel_size: List[float],
                 depth_samples: Tensor,
                 proj_mat: Tensor,
                 downsample_factor: int,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True) -> Tensor:
    """Obtain image features using points.

    Args:
        voxel_features (Tensor): 1 x C x Nx x Ny x Nz voxel features.
        voxel_range (List[float]): The range of voxel features.
        voxel_size (List[float]): The voxel size of voxel features.
        depth_samples (Tensor): N depth samples in LiDAR coordinates.
        proj_mat (Tensor): ORIGINAL LiDAR2img projection matrix for N views.
        downsample_factor (int): The downsample factor in rescaling.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        Tensor: 1xCxDxHxW frustum features sampled from voxel features.
    """
    # construct frustum grid
    device = voxel_features.device
    h, w = img_pad_shape
    h_out = round(h / downsample_factor)
    w_out = round(w / downsample_factor)
    ws = (torch.linspace(0, w_out - 1, w_out) * downsample_factor).to(device)
    hs = (torch.linspace(0, h_out - 1, h_out) * downsample_factor).to(device)
    depths = depth_samples[::downsample_factor]
    num_depths = len(depths)
    ds_3d, ys_3d, xs_3d = torch.meshgrid(depths, hs, ws)
    # grid: (D, H_out, W_out, 3) -> (D*H_out*W_out, 3)
    grid = torch.stack([xs_3d, ys_3d, ds_3d], dim=-1).view(-1, 3)
    # recover the coordinates in the canonical space
    # reverse order of augmentations: flip -> crop -> scale
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        grid[:, 0] = ori_w - grid[:, 0]
    grid[:, :2] += img_crop_offset
    grid[:, :2] /= img_scale_factor
    # grid3d: (D*H_out*W_out, 3) in LiDAR coordinate system
    grid3d = points_img2cam(grid, proj_mat)
    # convert the 3D point coordinates to voxel coordinates
    voxel_range = torch.tensor(voxel_range).to(device).view(1, 6)
    voxel_size = torch.tensor(voxel_size).to(device).view(1, 3)
    # suppose the voxel grid is generated with AlignedAnchorGenerator
    # -0.5 given each grid is located at the center of the grid
    # TODO: study whether here needs -0.5
    grid3d = (grid3d - voxel_range[:, :3]) / voxel_size - 0.5
    grid_size = (voxel_range[:, 3:] - voxel_range[:, :3]) / voxel_size
    # normalize grid3d to (-1, 1)
    grid3d = grid3d / grid_size * 2 - 1
    # (x, y, z) -> (z, y, x) for grid_sampling
    grid3d = grid3d.view(1, num_depths, h_out, w_out, 3)[..., [2, 1, 0]]
    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    frustum_features = F.grid_sample(
        voxel_features,
        grid3d,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCxDxHxW feats

    return frustum_features
