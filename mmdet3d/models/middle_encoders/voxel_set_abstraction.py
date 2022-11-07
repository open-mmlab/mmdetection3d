# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.furthest_point_sample import furthest_point_sample
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from mmdet3d.utils import InstanceList


def bilinear_interpolate_torch(inputs, x, y):
    """Bilinear interpolate for inputs."""
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, inputs.shape[1] - 1)
    x1 = torch.clamp(x1, 0, inputs.shape[1] - 1)
    y0 = torch.clamp(y0, 0, inputs.shape[0] - 1)
    y1 = torch.clamp(y1, 0, inputs.shape[0] - 1)

    Ia = inputs[y0, x0]
    Ib = inputs[y1, x0]
    Ic = inputs[y0, x1]
    Id = inputs[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(
        torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


@MODELS.register_module()
class VoxelSetAbstraction(BaseModule):
    """Voxel set abstraction module for PVRCNN and PVRCNN++.

    Args:
        num_keypoints (int): The number of key points sampled from
            raw points cloud.
        fused_out_channel (int): Key points feature output channels
            num after fused. Default to 128.
        voxel_size (list[float]): Size of voxels. Defaults to
            [0.05, 0.05, 0.1].
        point_cloud_range (list[float]): Point cloud range. Defaults to
            [0, -40, -3, 70.4, 40, 1].
        voxel_sa_cfgs_list (List[dict or ConfigDict], optional): List of SA
            module cfg. Used to gather key points features from multi-wise
            voxel features. Default to None.
        rawpoints_sa_cfgs (dict or ConfigDict, optional): SA module cfg.
            Used to gather key points features from raw points. Default to
            None.
        bev_feat_channel (int): Bev features channels num.
            Default to 256.
        bev_scale_factor (int): Bev features scale factor. Default to 8.
        voxel_center_as_source (bool): Whether used voxel centers as points
            cloud key points. Defaults to False.
        norm_cfg (dict[str]): Config of normalization layer. Default
            used dict(type='BN1d', eps=1e-5, momentum=0.1).
        bias (bool | str, optional): If specified as `auto`, it will be
            decided by `norm_cfg`. `bias` will be set as True if
            `norm_cfg` is None, otherwise False. Default: 'auto'.
    """

    def __init__(self,
                 num_keypoints: int,
                 fused_out_channel: int = 128,
                 voxel_size: list = [0.05, 0.05, 0.1],
                 point_cloud_range: list = [0, -40, -3, 70.4, 40, 1],
                 voxel_sa_cfgs_list: Optional[list] = None,
                 rawpoints_sa_cfgs: Optional[dict] = None,
                 bev_feat_channel: int = 256,
                 bev_scale_factor: int = 8,
                 voxel_center_as_source: bool = False,
                 norm_cfg: dict = dict(type='BN2d', eps=1e-5, momentum=0.1),
                 bias: str = 'auto') -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.fused_out_channel = fused_out_channel
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_center_as_source = voxel_center_as_source

        gathered_channel = 0

        if rawpoints_sa_cfgs is not None:
            self.rawpoints_sa_layer = MODELS.build(rawpoints_sa_cfgs)
            gathered_channel += sum(
                [x[-1] for x in rawpoints_sa_cfgs.mlp_channels])
        else:
            self.rawpoints_sa_layer = None

        if voxel_sa_cfgs_list is not None:
            self.voxel_sa_configs_list = voxel_sa_cfgs_list
            self.voxel_sa_layers = nn.ModuleList()
            for voxel_sa_config in voxel_sa_cfgs_list:
                cur_layer = MODELS.build(voxel_sa_config)
                self.voxel_sa_layers.append(cur_layer)
                gathered_channel += sum(
                    [x[-1] for x in voxel_sa_config.mlp_channels])
        else:
            self.voxel_sa_layers = None

        if bev_feat_channel is not None and bev_scale_factor is not None:
            self.bev_cfg = mmengine.Config(
                dict(
                    bev_feat_channels=bev_feat_channel,
                    bev_scale_factor=bev_scale_factor))
            gathered_channel += bev_feat_channel
        else:
            self.bev_cfg = None
        self.point_feature_fusion_layer = nn.Sequential(
            ConvModule(
                gathered_channel,
                fused_out_channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                bias=bias))

    def interpolate_from_bev_features(self, keypoints: torch.Tensor,
                                      bev_features: torch.Tensor,
                                      batch_size: int,
                                      bev_scale_factor: int) -> torch.Tensor:
        """Gather key points features from bev feature map by interpolate.

        Args:
            keypoints (torch.Tensor): Sampled key points with shape
                (N1 + N2 + ..., NDim).
            bev_features (torch.Tensor): Bev feature map from the first
                stage with shape (B, C, H, W).
            batch_size (int): Input batch size.
            bev_scale_factor (int): Bev feature map scale factor.

        Returns:
            torch.Tensor: Key points features gather from bev feature
                map with shape (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[..., 0] -
                  self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[..., 1] -
                  self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_scale_factor
        y_idxs = y_idxs / bev_scale_factor

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k, ...]
            cur_y_idxs = y_idxs[k, ...]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(
                cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(
            point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features.view(batch_size, keypoints.shape[1], -1)

    def get_voxel_centers(self, coors: torch.Tensor,
                          scale_factor: float) -> torch.Tensor:
        """Get voxel centers coordinate.

        Args:
            coors (torch.Tensor): Coordinates of voxels shape is Nx(1+NDim),
                where 1 represents the batch index.
            scale_factor (float): Scale factor.

        Returns:
            torch.Tensor: Voxel centers coordinate with shape (N, 3).
        """
        assert coors.shape[1] == 4
        voxel_centers = coors[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = torch.tensor(
            self.voxel_size,
            device=voxel_centers.device).float() * scale_factor
        pc_range = torch.tensor(
            self.point_cloud_range[0:3], device=voxel_centers.device).float()
        voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
        return voxel_centers

    def sample_key_points(self, points: List[torch.Tensor],
                          coors: torch.Tensor) -> torch.Tensor:
        """Sample key points from raw points cloud.

        Args:
            points (List[torch.Tensor]): Point cloud of each sample.
            coors (torch.Tensor): Coordinates of voxels shape is Nx(1+NDim),
                where 1 represents the batch index.

        Returns:
            torch.Tensor: (B, M, 3) Key points of each sample.
                M is num_keypoints.
        """
        assert points is not None or coors is not None
        if self.voxel_center_as_source:
            _src_points = self.get_voxel_centers(coors=coors, scale_factor=1)
            batch_size = coors[-1, 0].item() + 1
            src_points = [
                _src_points[coors[:, 0] == b] for b in range(batch_size)
            ]
        else:
            src_points = [p[..., :3] for p in points]

        keypoints_list = []
        for points_to_sample in src_points:
            num_points = points_to_sample.shape[0]
            cur_pt_idxs = furthest_point_sample(
                points_to_sample.unsqueeze(dim=0).contiguous(),
                self.num_keypoints).long()[0]

            if num_points < self.num_keypoints:
                times = int(self.num_keypoints / num_points) + 1
                non_empty = cur_pt_idxs[:num_points]
                cur_pt_idxs = non_empty.repeat(times)[:self.num_keypoints]

            keypoints = points_to_sample[cur_pt_idxs]

            keypoints_list.append(keypoints)
        keypoints = torch.stack(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, batch_inputs_dict: dict, feats_dict: dict,
                rpn_results_list: InstanceList) -> dict:
        """Extract point-wise features from multi-input.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.
            feats_dict (dict): Contains features from the first
                stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.

        Returns:
            dict: Contain Point-wise features, include:
                - keypoints (torch.Tensor): Sampled key points.
                - keypoint_features (torch.Tensor): Gathered key points
                    features from multi input.
                - fusion_keypoint_features (torch.Tensor): Fusion
                    keypoint_features by point_feature_fusion_layer.
        """
        points = batch_inputs_dict['points']
        voxel_encode_features = feats_dict['multi_scale_3d_feats']
        bev_encode_features = feats_dict['spatial_feats']
        if self.voxel_center_as_source:
            voxels_coors = batch_inputs_dict['voxels']['coors']
        else:
            voxels_coors = None
        keypoints = self.sample_key_points(points, voxels_coors)

        point_features_list = []
        batch_size = len(points)

        if self.bev_cfg is not None:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, bev_encode_features, batch_size,
                self.bev_cfg.bev_scale_factor)
            point_features_list.append(point_bev_features.contiguous())

        batch_size, num_keypoints, _ = keypoints.shape
        key_xyz = keypoints.view(-1, 3)
        key_xyz_batch_cnt = key_xyz.new_zeros(batch_size).int().fill_(
            num_keypoints)

        if self.rawpoints_sa_layer is not None:
            batch_points = torch.cat(points, dim=0)
            batch_cnt = [len(p) for p in points]
            xyz = batch_points[:, :3].contiguous()
            features = None
            if batch_points.size(1) > 0:
                features = batch_points[:, 3:].contiguous()
            xyz_batch_cnt = xyz.new_tensor(batch_cnt, dtype=torch.int32)

            pooled_points, pooled_features = self.rawpoints_sa_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=key_xyz.contiguous(),
                new_xyz_batch_cnt=key_xyz_batch_cnt,
                features=features.contiguous(),
            )

            point_features_list.append(pooled_features.contiguous().view(
                batch_size, num_keypoints, -1))
        if self.voxel_sa_layers is not None:
            for k, voxel_sa_layer in enumerate(self.voxel_sa_layers):
                cur_coords = voxel_encode_features[k].indices
                xyz = self.get_voxel_centers(
                    coors=cur_coords,
                    scale_factor=self.voxel_sa_configs_list[k].scale_factor
                ).contiguous()
                xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

                pooled_points, pooled_features = voxel_sa_layer(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=key_xyz.contiguous(),
                    new_xyz_batch_cnt=key_xyz_batch_cnt,
                    features=voxel_encode_features[k].features.contiguous(),
                )
                point_features_list.append(pooled_features.contiguous().view(
                    batch_size, num_keypoints, -1))

        point_features = torch.cat(
            point_features_list, dim=-1).view(batch_size * num_keypoints, -1,
                                              1)

        fusion_point_features = self.point_feature_fusion_layer(
            point_features.unsqueeze(dim=-1)).squeeze(dim=-1)

        batch_idxs = torch.arange(
            batch_size * num_keypoints, device=keypoints.device
        ) // num_keypoints  # batch indexes of each key points
        batch_keypoints_xyz = torch.cat(
            (batch_idxs.to(key_xyz.dtype).unsqueeze(dim=-1), key_xyz), dim=-1)

        return dict(
            keypoint_features=point_features.squeeze(dim=-1),
            fusion_keypoint_features=fusion_point_features.squeeze(dim=-1),
            keypoints=batch_keypoints_xyz)
