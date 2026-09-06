# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import torch
import torchvision.models.resnet as resnet
from mmcv.cnn import build_conv_layer
from torch import nn

from mmdet3d.registry import MODELS
from ..utils.encoder_utils import I2P, ConvBNReLU, LocalContextAttentionBlock


class FocalEncoderLayer(nn.Module):
    """Single encoder layer for FocalFormer3D multi-modal BEV fusion.

    Supports multiple fusion strategies:
      - 'bevfusion': I2P cross-attention + local context attention
      - 'bevfusionmb2': I2P cross-attention + MobileNetV2 inverted residuals
      - other: simple convolution on BEV features
    """

    def __init__(self,
                 hidden_channel,
                 iterbev='bevfusion',
                 max_points_height=5,
                 iterbev_wo_img=False,
                 multiscale_outputs=False,
                 layer_id=None,
                 iter_bev_cam=None,
                 need_projbev=True):
        super(FocalEncoderLayer, self).__init__()

        self.iterbev = iterbev
        self.iterbev_wo_img = iterbev_wo_img
        self.multiscale_outputs = multiscale_outputs
        self.layer_id = layer_id
        self.iter_bev_cam = iter_bev_cam
        self.need_projbev = need_projbev

        # --- I2P (Image-to-Point) block ---
        if self.iterbev in ['bevfusion', 'bevfusionmb2']:
            if need_projbev and (not self.iter_bev_cam or self.layer_id == 0):
                if not self.iterbev_wo_img:
                    self.I2P_block = I2P(
                        hidden_channel,
                        hidden_channel,
                        0.1,
                        max_points_height=max_points_height)
            else:
                self.I2P_block = None

        # --- BEV fusion backbone per strategy ---
        if self.iterbev == 'bevfusionmb2':
            import torchvision.models.mobilenetv2 as mobilenetv2
            self.P_IML = mobilenetv2.InvertedResidual(
                hidden_channel,
                hidden_channel,
                stride=1,
                expand_ratio=2,
                norm_layer=nn.BatchNorm2d)
            self.P_out_proj = mobilenetv2.InvertedResidual(
                2 * hidden_channel,
                hidden_channel,
                stride=1,
                expand_ratio=1,
                norm_layer=nn.BatchNorm2d)
            self.P_integration = mobilenetv2.InvertedResidual(
                2 * hidden_channel,
                hidden_channel,
                stride=1,
                expand_ratio=1,
                norm_layer=nn.BatchNorm2d)

        elif self.iterbev == 'bevfusion':
            self.P_IML = LocalContextAttentionBlock(hidden_channel,
                                                    hidden_channel, 9)
            self.P_out_proj = ConvBNReLU(
                2 * hidden_channel,
                hidden_channel,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None)
            self.P_integration = ConvBNReLU(
                2 * hidden_channel,
                hidden_channel,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None)

        else:
            self.iterbev_conv = ConvBNReLU(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None)

        # --- Image iteration conv ---
        if self.iterbev_wo_img:
            self.iterimg_conv = None
        else:
            self.iterimg_conv = nn.Sequential(
                resnet.BasicBlock(
                    hidden_channel, hidden_channel,
                    norm_layer=nn.BatchNorm2d), )

    def forward(self, img_feat, lidar_feat, img_metas, extra_args=None):
        """Forward pass for a single FocalEncoder layer.

        Args:
            img_feat (torch.Tensor or None): Image features [BN, C, H, W].
            lidar_feat (torch.Tensor): LiDAR BEV features [B, C, H', W'].
            img_metas (list[dict]): Image metadata.
            extra_args (dict, optional): Extra arguments. Defaults to None.

        Returns:
            tuple: (new_img_feat, new_lidar_feat)
        """
        batch_size = lidar_feat.shape[0]

        if not self.iterbev_wo_img:
            BN, I_C, I_H, I_W = img_feat.shape

        # --- Cross-modal feature extraction ---
        if self.iterbev in ['bevfusion', 'bevfusionmb2']:
            if not self.iterbev_wo_img:
                if self.iter_bev_cam:
                    if self.layer_id == 0 and self.need_projbev:
                        I2P_feat = self.I2P_block(
                            lidar_feat,
                            img_feat.view(batch_size, -1, I_C, I_H, I_W),
                            img_metas)
                        img_feat = I2P_feat
                    else:
                        I2P_feat = img_feat
                else:
                    I2P_feat = self.I2P_block(
                        lidar_feat, img_feat.view(batch_size, -1, I_C, I_H,
                                                  I_W), img_metas)
            else:
                I2P_feat = lidar_feat

        # --- BEV feature fusion ---
        if self.iterbev == 'bevfusion':
            P2P_feat = self.P_IML(lidar_feat, lidar_feat)
            P_Aug_feat = self.P_out_proj(
                torch.cat((I2P_feat, P2P_feat), dim=1))
            new_lidar_feat = self.P_integration(
                torch.cat((P_Aug_feat, lidar_feat), dim=1))

        elif self.iterbev == 'bevfusionmb2':
            P2P_feat = self.P_IML(lidar_feat)
            P_Aug_feat = self.P_out_proj(
                torch.cat((I2P_feat, P2P_feat), dim=1))
            new_lidar_feat = self.P_integration(
                torch.cat((P_Aug_feat, lidar_feat), dim=1))

        else:
            new_lidar_feat = self.iterbev_conv(lidar_feat)

        # --- Image feature iteration ---
        if not self.iterimg_conv:
            new_img_feat = None
        else:
            new_img_feat = self.iterimg_conv(img_feat)

        return new_img_feat, new_lidar_feat


@MODELS.register_module()
class FocalEncoder(nn.Module):
    """FocalFormer3D encoder neck for multi-modal BEV fusion.

    Applies shared convolutions (or LSS camera projection) followed by N layers
    of iterative BEV fusion between image and point cloud features. Supports
    multi-stage heatmap outputs for hard instance probing.
    """

    def __init__(
        self,
        num_layers=2,
        in_channels_img=64,
        in_channels_pts=128 * 3,
        hidden_channel=128,
        bn_momentum=0.1,
        bias='auto',
        iterbev='bevfusion',
        max_points_height=5,
        multistage_heatmap=False,
        input_img=True,
        input_pts=True,
        iterbev_wo_img=False,
        extra_feat=False,
        iter_bev_cam=False,
        cam_lss=False,
        newbevpool=False,
        pc_range=None,
        img_scale=None,
    ):
        super(FocalEncoder, self).__init__()

        self.iterbev_wo_img = iterbev_wo_img
        self.iterbev = iterbev
        self.iter_bev_cam = iter_bev_cam

        self.multistage_heatmap = multistage_heatmap

        # --- Point cloud shared conv ---
        self.input_pts = input_pts
        if self.input_pts:
            self.shared_conv_pts = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_pts,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

        # --- Image shared conv or LSS camera projection ---
        self.input_img = input_img
        self.cam_proj_type = cam_lss
        if self.input_img:
            if cam_lss:
                from .lss import LiftSplatShoot
                self.cam_lss = LiftSplatShoot(
                    grid=0.6,
                    inputC=256,
                    outputC=hidden_channel,
                    camC=64,
                    pc_range=pc_range,
                    img_scale=img_scale,
                    downsample=4,
                    newbevpool=newbevpool)
            else:
                self.cam_lss = None
                self.shared_conv_img = build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels_img,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                )

        # --- Fusion layers ---
        # num_layers=None means no fusion blocks at all, which is how
        # DeformFormer3D reduces this neck to shared_conv_pts alone.
        self.num_layers = num_layers if num_layers else 0
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusion_blocks.append(
                FocalEncoderLayer(
                    hidden_channel,
                    iterbev=iterbev,
                    max_points_height=max_points_height,
                    iterbev_wo_img=self.iterbev_wo_img,
                    multiscale_outputs=False,
                    layer_id=i,
                    iter_bev_cam=iter_bev_cam,
                    need_projbev=not cam_lss), )

        # --- Extra feature output ---
        self.extra_feat = extra_feat
        if self.extra_feat:
            self.extra_output = ConvBNReLU(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None)

        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas):
        """Forward pass.

        Args:
            img_feats (torch.Tensor or None): Image features.
            pts_feats (torch.Tensor or None): Point cloud BEV features.
            img_metas (list[dict]): Image metadata.

        Returns:
            tuple: (new_img_feat, [pts_feat_conv, new_pts_feat])
        """
        batch_size = len(img_metas)

        # ---- Image feature projection ----
        if self.input_img:
            if self.cam_proj_type:
                if self.cam_proj_type == 'proj':
                    new_img_feat = self.cam_lss(
                        img_feats.view(batch_size, -1, *img_feats.shape[-3:]),
                        img_metas=img_metas)
                else:
                    rots = []
                    trans = []
                    for sample_idx in range(batch_size):
                        rot_list = []
                        trans_list = []
                        for mat in img_metas[sample_idx]['lidar2img']:
                            mat = torch.Tensor(mat).cuda()
                            inverse_mat = mat.inverse()
                            rot_list.append(inverse_mat[:3, :3])
                            trans_list.append(inverse_mat[:3, 3].view(-1))
                        rot_list = torch.stack(rot_list, dim=0)
                        trans_list = torch.stack(trans_list, dim=0)
                        rots.append(rot_list)
                        trans.append(trans_list)
                    rots = torch.stack(rots)
                    trans = torch.stack(trans)
                    new_img_feat, depth = self.cam_lss(
                        img_feats.view(batch_size, -1, *img_feats.shape[-3:]),
                        rots=rots,
                        trans=trans,
                        img_metas=img_metas)

                if not self.input_pts and not self.multistage_heatmap:
                    return None, [new_img_feat, new_img_feat]
            else:
                new_img_feat = self.shared_conv_img(img_feats)
        else:
            new_img_feat = None

        # ---- Point cloud feature projection ----
        if self.input_pts:
            new_pts_feat = self.shared_conv_pts(pts_feats)
        else:
            # set empty for train image only (nuscenes bev feature size)
            new_pts_feat = torch.zeros((batch_size, 128, 180, 180),
                                       device='cuda')

        pts_feat_conv = new_pts_feat.clone()

        # ---- Iterative fusion ----
        if self.input_img or self.iterbev_wo_img:
            multistage_ptsfeats = []
            extra_args = {}

            for i in range(self.num_layers):
                new_img_feat, new_pts_feat = self.fusion_blocks[i](
                    new_img_feat, new_pts_feat, img_metas, extra_args)

                if self.multistage_heatmap:
                    multistage_ptsfeats.append(new_pts_feat)

            if self.multistage_heatmap:
                new_pts_feat = multistage_ptsfeats
                if self.extra_feat:
                    new_pts_feat.append(self.extra_output(new_pts_feat[-1]))

            return new_img_feat, [pts_feat_conv, new_pts_feat]
        else:
            return None, [new_pts_feat, None]
