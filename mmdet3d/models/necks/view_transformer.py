# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


class QuickCumsum(torch.autograd.Function):
    """Back propagation accelerated cumulative sum operator."""

    @staticmethod
    def forward(ctx, x, coor, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, coor = x[kept], coor[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for coordinates
        ctx.mark_non_differentiable(coor)

        return x, coor

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@NECKS.register_module()
class ViewTransformerLiftSplatShoot(BaseModule):
    """Lift-Splat-Shoot view transformer for transform image-view feature into
    bird-eye-view feature.

    Args:
        grid_config (dict(axis:list(float,float,float))): Config of grid alone
            each axis in format of (lower_bound, upper_bound, interval). axis
            in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        in_channels (int): Channels of input feature.
        tran_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        max_voxel_points (int): Specify the maximum point number in a single
            voxel during the acceleration.
    """

    def __init__(self,
                 grid_config,
                 input_size,
                 downsample,
                 in_channels,
                 tran_channels,
                 accelerate=False,
                 max_voxel_points=300):
        super(ViewTransformerLiftSplatShoot, self).__init__()
        self.grid_config = grid_config
        self.gen_grid_infos(**self.grid_config)

        self.input_size = input_size
        self.downsample = downsample

        self.create_frustum()
        self.tran_channels = tran_channels
        self.depthnet = nn.Conv2d(
            in_channels, self.D + self.tran_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.max_voxel_points = max_voxel_points
        self.initial_flag = True

    def gen_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x: Config of grid alone x axis in format of (lower_bound,
                upper_bound, interval).
            y: Config of grid alone y axis in format of (lower_bound,
                upper_bound, interval).
            z: Config of grid alone z axis in format of (lower_bound,
                upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = \
            nn.Parameter(torch.Tensor([cfg[0] + cfg[2]/2.0
                                       for cfg in [x, y, z]]),
                         requires_grad=False)
        self.grid_interval = \
            nn.Parameter(torch.Tensor([cfg[2] for
                                       cfg in [x, y, z]]),
                         requires_grad=False)
        self.grid_size = \
            nn.Parameter(torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]]),
                         requires_grad=False)

    def create_frustum(self):
        """Generate the frustum template for each image."""
        H_in, W_in = self.input_size
        H_feat, W_feat = H_in // self.downsample, W_in // self.downsample
        d = torch.arange(*self.grid_config['depth'], dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        frustum = torch.stack((x, y, d), -1)
        self.frustum = nn.Parameter(frustum, requires_grad=False)

    def get_lidar_coor(self, rots, trans, intrins, post_rots, post_trans):
        """Calculate the locations of the frustum points in the lidar(ego)
        coordinate system.

        Args:
            rots (torch.Tensor): of shape (N, N_cams, 3, 3). Rotation from
                camera coordinate system to lidar coordinate system.
            trans (torch.Tensor): of shape (N, N_cams, 3). Translation from
                camera coordinate system to lidar coordinate system.
            intrins (torch.Tensor): of shape (N, N_cams, 3, 3). Camera
                intrinsic matrixes.
            post_rots (torch.Tensor): of shape (N, N_cams, 3, 3). Rotation in
                camera coordinate system derived from image view augmentation.
            post_trans (torch.Tensor): of shape (N, N_cams, 3). Translation in
                camera coordinate system derived from image view augmentation.

        Returns: Points with size B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def voxel_pooling_prepare(self, coor, x):
        """Data preparation for voxel pooling."""
        B, N, D, H, W, C = x.shape
        num_points = B * N * D * H * W
        # flatten x
        x = x.reshape(num_points, C)
        # record the index of selected points for acceleration purpose
        point_idx = torch.range(0, num_points - 1, dtype=torch.long)
        # flatten indices
        coor = ((coor - (self.grid_lower_bound - self.grid_interval / 2.)) /
                self.grid_interval).long()
        coor = coor.view(num_points, 3)
        batch_idx = torch.cat([
            torch.full([num_points // B, 1],
                       idx,
                       device=x.device,
                       dtype=torch.long) for idx in range(B)
        ])
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        x, coor, point_idx = x[kept], coor[kept], point_idx[kept]

        # get tensors from the same voxel next to each other
        ranks = coor[:, 0] * (self.grid_size[1] * self.grid_size[2] * B)
        ranks += coor[:, 1] * (self.grid_size[2] * B)
        ranks += coor[:, 2] * B + coor[:, 3]
        sorts = ranks.argsort()
        return x[sorts], coor[sorts], ranks[sorts], point_idx[sorts]

    def voxel_pooling(self, coor, x):
        """Generate bird-eye-view features with the pseudo point cloud.

        Args:
            coor (torch.tensor): of shape (B, N, D,  H, W, 3). Coordinate
                of points.
            x (torch.tensor): of shape (B, N, D,  H, W, C). Feature of points.

        Returns: Bird-eye-view features of shape (B, C, H_BEV, W_BEV).
        """
        B, N, D, H, W, C = x.shape
        x, coor, ranks, _ = self.voxel_pooling_prepare(coor, x)

        # cumsum trick
        x, coor = QuickCumsum.apply(x, coor, ranks)

        # griddify (B x C x Z x X x Y)
        grid_size = self.grid_size.to(torch.long)
        final = torch.zeros((B, C, grid_size[2], grid_size[1], grid_size[0]),
                            device=x.device)
        final[coor[:, 3], :, coor[:, 2], coor[:, 1], coor[:, 0]] = x
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def acceleration_initialize(self, coor, x):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): of shape (B, N, D,  H, W, 3). Coordinate
                of points.
            x (torch.tensor): of shape (B, N, D,  H, W, C). Feature of points.
        """
        x, coor, ranks, point_idx = self.voxel_pooling_prepare(coor, x)

        repeat_times = torch.ones(
            coor.shape[0], device=coor.device, dtype=coor.dtype)
        times = 0
        repeat_times[0] = 0
        curr_rank = ranks[0]

        for i in range(1, ranks.shape[0]):
            if curr_rank == ranks[i]:
                times += 1
                repeat_times[i] = times
            else:
                curr_rank = ranks[i]
                times = 0
                repeat_times[i] = times
        kept = repeat_times < self.max_voxel_points
        repeat_times, coor = repeat_times[kept], coor[kept]
        x, point_idx = x[kept], point_idx[kept]

        coor = torch.cat([coor, repeat_times.unsqueeze(-1)], dim=-1)
        self.coor = coor
        self.point_idx = point_idx
        self.initial_flag = False

    def voxel_pooling_accelerated(self, x):
        """Conducting voxel pooling in accelerated mode.

        Args:
            x (torch.tensor): of shape (B, N, D, H, W, C). The feature of the
                volumes.

        Returns: Bird-eye-view features of shape (B, C, H_BEV, W_BEV).
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)[self.point_idx]

        # griddify (B x C x Z x X x Y)
        gs = self.grid_size.to(torch.long)
        final = torch.zeros((B, C, gs[2], gs[1], gs[0], self.max_voxel_points),
                            device=x.device)
        c = self.coor
        final[c[:, 3], :, c[:, 2], c[:, 1], c[:, 0], c[:, 4]] = x
        final = final.sum(-1)

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns: bird-eye-view feature of shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = x[:, :self.D]
        depth = depth.softmax(dim=1)
        tran_feat = x[:, self.D:(self.D + self.tran_channels)]

        # Lift
        volume = depth.unsqueeze(1) * tran_feat.unsqueeze(2)
        volume = volume.view(B, N, self.tran_channels, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            if self.initial_flag:
                coor = self.get_lidar_coor(*input[1:])
                self.acceleration_initialize(coor, volume)
            bev_feat = self.voxel_pooling_accelerated(volume)
        else:
            coor = self.get_lidar_coor(*input[1:])
            bev_feat = self.voxel_pooling(coor, volume)
        return bev_feat
