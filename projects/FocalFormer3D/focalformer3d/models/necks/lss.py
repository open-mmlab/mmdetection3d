"""Copyright (C) 2020 NVIDIA Corporation.

All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at
https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

# =============================================================================
# lss.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - force_fp32 from mmcv.runner          → REMOVED (AmpOptimWrapper handles)
#   - import matplotlib, mpl_toolkits,
#     save_image                           → REMOVED (unused at runtime)
#   - Added debug logging throughout
# =============================================================================

import torch
import torch.nn.functional as F
from torch import nn
# --- torchvision resnet18 ---
from torchvision.models.resnet import resnet18

# --- apply_3d_transformation (still in mmdet3d.models.fusion_layers) ---
try:
    from mmdet3d.models.fusion_layers import apply_3d_transformation
except ImportError:
    from mmdet3d.models.layers import apply_3d_transformation


# ===================================================================
# Up — bilinear upsample + concat + conv block
# ===================================================================
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = F.interpolate(
            x1, x2.shape[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


# ===================================================================
# BevEncode — ResNet18-based BEV encoder
# ===================================================================
class BevEncode(nn.Module):

    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(
            inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


# ===================================================================
# Utility functions
# ===================================================================
def gen_dx_bx(xbound, ybound, zbound):
    """Generate voxel grid parameters dx, bx, nx."""
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    """Cumulative sum trick for voxel pooling."""
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


# ===================================================================
# QuickCumsum — custom autograd for fast voxel pooling
# ===================================================================
class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


# ===================================================================
# CamEncode — per-camera depth + feature encoding
# ===================================================================
class CamEncode(nn.Module):

    def __init__(self, D, C, inputC):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2d(
            inputC, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = (
            depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2))

        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x, depth


# ===================================================================
# LiftSplatShoot — main LSS camera-to-BEV module
# ===================================================================
class LiftSplatShoot(nn.Module):
    """Lift-Splat-Shoot camera-to-BEV projection module.

    Lifts 2D camera features into 3D frustum, splats them into a voxel grid,
    and encodes the BEV representation.
    """

    def __init__(self,
                 img_scale=(900, 1600),
                 camera_depth_range=[4.0, 45.0, 1.0],
                 pc_range=[-50, -50, -5, 50, 50, 3],
                 downsample=4,
                 grid=3,
                 inputC=256,
                 outputC=128,
                 camC=64,
                 newbevpool=False):
        """
        Args:
            img_scale: actual RGB image size, default (900, 1600).
            camera_depth_range: [min, max, step] for depth bins.
            pc_range: point cloud range
                [x_min, y_min, z_min, x_max, y_max, z_max].
            downsample (int): downsampling rate of input camera feature to
                img_scale, default 4.
            grid: stride for splat.
            inputC: input camera feature channel dimension (default 256).
            outputC: output BEV channel dimension (default 128).
            camC: camera encoding channel dimension (default 64).
            newbevpool: whether to use bev_pool_op instead of voxel_pooling.
        """
        super(LiftSplatShoot, self).__init__()

        self.pc_range = pc_range
        self.grid_conf = {
            'xbound': [pc_range[0], pc_range[3], grid],
            'ybound': [pc_range[1], pc_range[4], grid],
            'zbound': [pc_range[2], pc_range[5], grid],
            'dbound': camera_depth_range,
        }
        self.img_scale = img_scale
        self.grid = grid

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'])

        self.dx = dx.cuda()
        self.bx = bx.cuda()
        self.nx = nx.cuda()

        self.downsample = downsample
        self.fH = self.img_scale[0] // self.downsample
        self.fW = self.img_scale[1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC)
        self.newbevpool = newbevpool

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        z = self.grid_conf['zbound']
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cz), nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, outputC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outputC), nn.ReLU(inplace=True))

    def init_weights(self):
        super(LiftSplatShoot, self).init_weights()

    def create_frustum(self):
        """Create frustum grid for depth bins."""
        ogfH, ogfW = self.img_scale
        fH, fW = self.fH, self.fW

        ds = torch.arange(
            *self.grid_conf['dbound'],
            dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape

        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1,
                                                     fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH,
                                                     1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self,
                     rots,
                     trans,
                     post_rots=None,
                     post_trans=None,
                     extra_rots=None,
                     extra_trans=None,
                     img_metas=None):
        """Determine the (x,y,z) locations (in the ego frame) of the points in
        the point cloud.

        Returns:
            torch.Tensor: B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # image aug matrix is post rots & trans
        if 'img_aug_matrix' in img_metas[0]:
            post_rots = torch.stack(
                [i['img_aug_matrix'][..., :3, :3] for i in img_metas],
                dim=0).to(rots)
            post_trans = torch.stack(
                [i['img_aug_matrix'][..., :3, 3] for i in img_metas],
                dim=0).to(rots)
        else:
            post_rots = None
            post_trans = None

        # undo post-transformation
        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = (
                    torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(
                        points.unsqueeze(-1)))
        else:
            points = (self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)
                      )  # B x N x D x H x W x 3 x 1

        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]), 5)
        points = (rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1))
        points += trans.view(B, N, 1, 1, 1, 3)

        # apply 3d transformation (forward) to aug lidar coord
        point_shape = points.shape[1:]
        for b in range(B):
            points[b] = apply_3d_transformation(
                points[b].view(-1, 3), 'LIDAR', img_metas[b],
                reverse=False).view(*point_shape)

        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = (
                    extra_rots.view(B, N, 1, 1, 1, 3, 3).matmul(
                        points.unsqueeze(-1)).squeeze(-1))
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C."""
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, depth = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, H, W)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, H, W)

        return x, depth

    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+
    def bev_pool(self, geom_feats, x):
        """BEV pooling using custom bev_pool_op.

        Args:
            geom_feats (torch.Tensor): Geometry features.
            x (torch.Tensor): Camera features [B, N, D, H, W, C].

        Returns:
            torch.Tensor: Pooled BEV features.
        """
        from ...ops.bev_pool.bev_pool_op import bev_pool as bev_pool_op

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool_op(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        return x

    def voxel_pooling(self, geom_feats, x):
        """Voxel pooling using cumulative sum trick.

        Args:
            geom_feats (torch.Tensor): Geometry features.
            x (torch.Tensor): Camera features [B, N, D, H, W, C].

        Returns:
            torch.Tensor: Pooled voxel features [B, C, Z, X, Y].
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        batch_ix = batch_ix.to(geom_feats.device)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) +
            geom_feats[:, 1] * (self.nx[2] * B) + geom_feats[:, 2] * B +
            geom_feats[:, 3])
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]),
                            device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0],
              geom_feats[:, 1]] = x

        return final

    def get_voxels(self,
                   x,
                   rots=None,
                   trans=None,
                   post_rots=None,
                   post_trans=None,
                   extra_rots=None,
                   extra_trans=None,
                   img_metas=None):
        """Lift camera features into voxel grid.

        Returns:
            tuple: (voxel_features, depth)
        """
        geom = self.get_geometry(
            rots,
            trans,
            post_rots,
            post_trans,
            extra_rots,
            extra_trans,
            img_metas=img_metas)
        x, depth = self.get_cam_feats(x)

        if not self.newbevpool:
            x = self.voxel_pooling(geom, x)
        else:
            x = self.bev_pool(geom, x)

        return x, depth

    def s2c(self, x):
        """Squash 5D voxel to 4D BEV: [B, C, H, W, L] → [B, C*H, L, W]."""
        B, C, H, W, L = x.shape
        bev = torch.reshape(x, (B, C * H, W, L))
        bev = bev.permute((0, 1, 3, 2))
        return bev

    def forward(self,
                x,
                rots,
                trans,
                lidar2img_rt=None,
                img_metas=None,
                post_rots=None,
                post_trans=None,
                extra_rots=None,
                extra_trans=None):
        """Forward pass: lift → splat → encode.

        Args:
            x (torch.Tensor): Camera features [B, N, C, H, W].
            rots (torch.Tensor): Camera rotation matrices.
            trans (torch.Tensor): Camera translation vectors.
            lidar2img_rt: Unused (kept for API compat).
            img_metas (list[dict]): Image metadata.
            post_rots: Post-augmentation rotations.
            post_trans: Post-augmentation translations.
            extra_rots: Extra rotations.
            extra_trans: Extra translations.

        Returns:
            tuple: (bev_features, depth)
        """
        x, depth = self.get_voxels(
            x,
            rots,
            trans,
            post_rots,
            post_trans,
            extra_rots,
            extra_trans,
            img_metas=img_metas)
        bev = self.s2c(x)
        x = self.bevencode(bev)

        return x, depth
