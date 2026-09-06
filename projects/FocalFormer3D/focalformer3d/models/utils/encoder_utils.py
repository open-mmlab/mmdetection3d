# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- apply_3d_transformation ---
try:
    from mmdet3d.models.fusion_layers import apply_3d_transformation
except ImportError:
    from mmdet3d.models.layers import apply_3d_transformation

# --- Local CUDA ops for local attention (JIT-compiled on first import) ---
from ...ops import locatt_ops


# ===================================================================
# ConvBNReLU — 2D Conv + BatchNorm + ReLU block
# ===================================================================
class ConvBNReLU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU,
                 bias='auto',
                 inplace=True,
                 affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


# ===================================================================
# ConvBNReLU3D — 3D Conv + BatchNorm + ReLU block
# ===================================================================
class ConvBNReLU3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 padding=None,
                 norm_layer=nn.BatchNorm3d,
                 activation_layer=nn.ReLU,
                 bias='auto',
                 inplace=True,
                 affine=True):
        super().__init__()
        if not padding:
            padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


# ===================================================================
# similarFunction — custom autograd for local attention similarity
# ===================================================================
class similarFunction(torch.autograd.Function):
    """credit: https://github.com/zzd1992/Image-Local-Attention"""

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.similar_forward(
            x_ori, x_loc, kH, kW)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.similar_backward(
            x_loc, grad_outputs, kH, kW, True)
        grad_loc = locatt_ops.localattention.similar_backward(
            x_ori, grad_outputs, kH, kW, False)
        return grad_ori, grad_loc, None, None


# ===================================================================
# weightingFunction — custom autograd for local attention weighting
# ===================================================================
class weightingFunction(torch.autograd.Function):
    """credit: https://github.com/zzd1992/Image-Local-Attention"""

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.weighting_forward(
            x_ori, x_weight, kH, kW)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.weighting_backward_ori(
            x_weight, grad_outputs, kH, kW)
        grad_weight = locatt_ops.localattention.weighting_backward_weight(
            x_ori, grad_outputs, kH, kW)
        return grad_ori, grad_weight, None, None


# ===================================================================
# LocalContextAttentionBlock — local attention with custom CUDA ops
# ===================================================================
class LocalContextAttentionBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 last_affine=True,
                 in_channels_key=None):
        super().__init__()

        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        if in_channels_key is None:
            in_channels_key = in_channels

        self.kernel_size = kernel_size
        self.query_project = nn.Sequential(
            ConvBNReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU),
            ConvBNReLU(
                out_channels,
                out_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(
            ConvBNReLU(
                in_channels_key,
                out_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU),
            ConvBNReLU(
                out_channels,
                out_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(
            in_channels_key,
            out_channels,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_feats, source_feats, **kwargs):
        query = self.query_project(target_feats)
        key = self.key_project(source_feats)
        value = self.value_project(source_feats)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size,
                               self.kernel_size)

        return out


# ===================================================================
# Grid creation utilities
# ===================================================================
def create_2D_grid(x_size, y_size):
    """Create a 2D grid of coordinates on CUDA.

    Args:
        x_size (int): Grid size in x dimension.
        y_size (int): Grid size in y dimension.

    Returns:
        torch.Tensor: Grid coordinates [1, x_size*y_size, 2] on CUDA.
    """
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    batch_y, batch_x = torch.meshgrid(
        *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    batch_x = batch_x + 0.5
    batch_y = batch_y + 0.5
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
    coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)

    return coord_base.cuda()


def create_3D_grid(x_size, y_size, z_size):
    """Create a 3D grid of coordinates on CUDA.

    Args:
        x_size (int): Grid size in x dimension.
        y_size (int): Grid size in y dimension.
        z_size (int): Grid size in z dimension.

    Returns:
        torch.Tensor: Grid coordinates [1, x*y*z, 3] on CUDA.
    """
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size],
                [0, z_size - 1, z_size]]
    batch_z, batch_y, batch_x = torch.meshgrid(
        *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    batch_x = batch_x + 0.5
    batch_y = batch_y + 0.5
    batch_z = batch_z + 0.5
    coord_base = torch.cat([batch_x[None], batch_y[None], batch_z[None]],
                           dim=0)[None]
    coord_base = coord_base.view(1, 3, -1).permute(0, 2, 1)

    return coord_base.cuda()


# ===================================================================
# I2P — Image-to-Point cross-attention module
# ===================================================================
class I2P(nn.Module):
    """Image-to-Point (I2P) cross-attention for BEV fusion.

    Projects 3D voxel grid coordinates into camera views, samples image
    features via grid_sample, and fuses them with LiDAR BEV features using
    multi-head attention.
    """

    def __init__(self,
                 pts_channels,
                 img_channels,
                 dropout,
                 max_points_height=5):
        super().__init__()
        self.pts_channels = pts_channels
        self.img_channels = img_channels
        self.dropout = dropout
        self.max_points_height = max_points_height
        self.learnedAlign = nn.MultiheadAttention(
            pts_channels,
            1,
            dropout=dropout,
            kdim=img_channels,
            vdim=img_channels,
            batch_first=True)

    def forward(self, lidar_feat, img_feat, img_metas, **kwargs):
        """Forward pass for I2P cross-attention.

        Args:
            lidar_feat (torch.Tensor): LiDAR BEV features [B, C, H, W].
            img_feat (torch.Tensor): Multi-view image features
                [B, N, C, H_img, W_img].
            img_metas (list[dict]): Image metadata with 'lidar2img',
                'input_shape', optionally 'img_aug_matrix'.

        Returns:
            torch.Tensor: Decorated LiDAR features [B, C, H, W].
        """
        batch_size = len(img_metas)
        decorated_lidar_feat = torch.zeros_like(lidar_feat)

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = lidar_feat.new_tensor(lidar2img)  # (B, 6, 4, 4)

        if 'img_aug_matrix' in img_metas[0]:
            post_rots = torch.stack(
                [i['img_aug_matrix'][..., :3, :3] for i in img_metas],
                dim=0).cuda()
            post_trans = torch.stack(
                [i['img_aug_matrix'][..., :3, 3] for i in img_metas],
                dim=0).cuda()
        else:
            post_rots = None
            post_trans = None

        point_cloud_range = lidar_feat.new_tensor(
            [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        spatial_shape = [
            lidar_feat.shape[-1], lidar_feat.shape[-2], self.max_points_height
        ]
        grid_3d = (create_3D_grid(*(spatial_shape[::-1])) /
                   lidar_feat.new_tensor(spatial_shape))  # [d, w, h]
        grid_3d = (
            grid_3d * (point_cloud_range[3:] - point_cloud_range[:3]) +
            point_cloud_range[:3])
        grid_3d = grid_3d.squeeze()

        for b in range(batch_size):
            proj_mat = lidar2img[b]
            num_cam = proj_mat.shape[0]
            num_voxels, p_dim = grid_3d.shape
            pts = grid_3d.view(num_voxels, p_dim)[..., :3]

            voxel_pts = apply_3d_transformation(
                pts, 'LIDAR', img_metas[b], reverse=True).detach()
            voxel_pts = torch.cat(
                (voxel_pts, torch.ones_like(voxel_pts[..., :1])),
                dim=-1).unsqueeze(0).unsqueeze(-1)
            proj_mat = proj_mat.unsqueeze(1)
            xyz_cams = torch.matmul(proj_mat, voxel_pts).squeeze(-1)

            eps = 1e-5
            mask = (xyz_cams[..., 2:3] > eps)
            xy_cams = xyz_cams[..., 0:2] / torch.maximum(
                xyz_cams[..., 2:3],
                torch.ones_like(xyz_cams[..., 2:3]) * eps)

            if post_rots is not None or post_trans is not None:
                xy_cams = torch.cat(
                    [xy_cams,
                     xy_cams.new_ones(*xy_cams.shape[:-1], 1)],
                    dim=-1)
                xy_cams = (
                    post_rots[b].view(num_cam, 1, 3, 3).matmul(
                        xy_cams.unsqueeze(-1)).squeeze(-1) +
                    post_trans[b].view(num_cam, 1, 3))
                xy_cams = xy_cams[..., :2]

            img_shape = img_metas[b]['input_shape']
            xy_cams[..., 0] = xy_cams[..., 0] / img_shape[1]
            xy_cams[..., 1] = xy_cams[..., 1] / img_shape[0]
            xy_cams = (xy_cams - 0.5) * 2

            mask = (
                mask
                & (xy_cams[..., 0:1] > -1.0)
                & (xy_cams[..., 0:1] < 1.0)
                & (xy_cams[..., 1:2] > -1.0)
                & (xy_cams[..., 1:2] < 1.0))
            mask = torch.nan_to_num(mask)

            sampled_feat = F.grid_sample(img_feat[b],
                                         xy_cams.unsqueeze(-2)).squeeze(-1)

            mask = mask.view(num_cam, 1, *spatial_shape[::-1])
            sampled_feat = sampled_feat.view(num_cam, self.img_channels,
                                             *spatial_shape[::-1])

            # reduce multi-view images
            reduced_sampled_feat = ((sampled_feat * mask).sum(dim=0) /
                                    (mask.sum(dim=0) + 1e-10))
            reduced_sampled_feat = (
                reduced_sampled_feat.flatten(2, 3).transpose(0, 2))

            mask = ((mask[:, 0].sum(dim=0) > 0).view(
                -1, reduced_sampled_feat.shape[0]).t().unsqueeze(-1))

            K = reduced_sampled_feat
            V = reduced_sampled_feat
            Q = lidar_feat[b, :].flatten(1, 2).t().unsqueeze(1)

            valid = mask[..., 0].sum(dim=1) > 0
            attn_output = lidar_feat.new_zeros(
                num_voxels // self.max_points_height, 1, self.pts_channels)
            attn_output[valid] = self.learnedAlign(
                Q[valid],
                K[valid],
                V[valid],
                attn_mask=(~mask[valid]).permute(0, 2, 1))[0]

            decorated_lidar_feat[b, :] = (
                attn_output.squeeze(1).t().view(self.pts_channels,
                                                spatial_shape[-2],
                                                spatial_shape[-3]))

        return decorated_lidar_feat
