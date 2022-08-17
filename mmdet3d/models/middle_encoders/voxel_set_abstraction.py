# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_norm_layer
from mmcv.ops import PointsSampler, gather_points
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import InstanceData


@MODELS.register_module()
class VoxelSetAbstraction(BaseModule):
    """Voxel set abstraction module for PVRCNN and PVRCNN++.

    Args:
        keypoints_sampler (dict or ConfigDict): Key point sampler config.
            It is used to build `PointsSampler` to sample key points from
            raw points.
        voxel_size (list[float]): Size of voxels. Defaults to
            [0.05, 0.05, 0.1].
        point_cloud_range (list[float]): Point cloud range. Defaults to
            [0, -40, -3, 70.4, 40, 1].
        rawpoint_sa_cfg (dict or ConfigDict, optional): SA module cfg. Used to
            gather key points features from raw points. Default to None.
        voxel_sa_cfg_list (List[dict or ConfigDict], optional): List of SA
            module cfg. Used to gather key points features from multi-level
            voxel features. Default to None.
        bev_cfg (dict or ConfigDict, optional): Bev features encode cfg. Used
            to gather key points features from Bev features. Default to None.
        sample_mode (str): Key points sample mode include
            `raw_points` and `voxel_centers` modes. If used `raw_points`
            the module will use keypoints_sampler to gather key points from
            raw points. Else if used `voxel_centers`, the module will use
            voxel centers as key points to extract features. Default to
            `raw_points.`
        fused_out_channels (int): Key points feature output channel
            num after fused. Default to 128.
        norm_cfg (dict[str]): Config of normalization layer. Default
            used dict(type='BN1d', eps=1e-5, momentum=0.1)
    """

    def __init__(
        self,
        keypoints_sampler: dict,
        voxel_size: list = [0.05, 0.05, 0.1],
        point_cloud_range: list = [0, -40, -3, 70.4, 40, 1],
        rawpoint_sa_cfg: Optional[dict] = None,
        voxel_sa_cfg_list: Optional[List[dict]] = None,
        bev_cfg: Optional[dict] = None,
        sample_mode: str = 'raw_points',
        fused_out_channels: int = 128,
        norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1)
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        assert sample_mode in ['raw_points', 'voxel_centers']
        self.sample_model = sample_mode

        self.voxel_sa_cfg_list = voxel_sa_cfg_list
        self.rawpoint_sa_cfg = rawpoint_sa_cfg
        self.bev_cfg = bev_cfg

        self.keypoints_sampler = PointsSampler(**keypoints_sampler)

        in_channels = 0
        if self.bev_cfg is not None:
            in_channels += self.bev_cfg.in_channels

        if self.rawpoint_sa_cfg is not None:
            self.rawpoints_sa_layer = build_sa_module(self.rawpoint_sa_cfg)
            in_channels += sum(
                [x[-1] for x in self.rawpoints_sa_layer.mlp_channels])

        if self.voxel_sa_cfg_list is not None:
            self.voxel_sa_layers = ModuleList()
            self.downsample_scale_factor = []
            for idx, voxel_sa_cfg in enumerate(self.voxel_sa_cfg_list):
                self.downsample_scale_factor.append(
                    voxel_sa_cfg.pop('scale_factor'))
                cur_layer = build_sa_module(voxel_sa_cfg)
                self.voxel_sa_layers.append(cur_layer)
                in_channels += sum([x[-1] for x in cur_layer.mlp_channels])

        self.point_feature_fusion = nn.Sequential(
            nn.Linear(in_channels, fused_out_channels, bias=False),
            build_norm_layer(norm_cfg, fused_out_channels)[1], nn.ReLU())

    def interpolate_from_bev_features(self, keypoints: Tensor,
                                      bev_features: Tensor,
                                      scale_factor: int) -> Tensor:
        """Interpolate from Bev features by key points.

        Args:
            keypoints (torch.Tensor): Sampled key points from raw points,
                shape is (B,N,3)，where N is key points num.
            bev_features (torch.Tensor): Bev features, shape is (B,H,W,C).
            scale_factor (int): Down sample scale factor.

        Returns:
            torch.Tensor: Key points Bev features.
        """
        # TODO: Here is different from OpenPCDet in inference,
        # but I think it doesn't influence trained model accuracy.
        _, _, y_grid, x_grid = bev_features.shape

        voxel_size_xy = keypoints.new_tensor(self.voxel_size[:2])

        # top-left coors in bev view
        bev_tl_grid_cxy = keypoints.new_tensor(self.point_cloud_range[:2])
        # below-right coors in bev view
        bev_br_grid_cxy = keypoints.new_tensor(self.point_cloud_range[3:5])
        bev_tl_grid_cxy.add_(0.5 * voxel_size_xy * scale_factor)
        bev_br_grid_cxy.sub_(0.5 * voxel_size_xy * scale_factor)

        xy = keypoints[..., :2]

        grid_sample_xy = (xy - bev_tl_grid_cxy[None, None, :]) / (
            (bev_br_grid_cxy - bev_tl_grid_cxy)[None, None, :])

        grid_sample_xy = (grid_sample_xy * 2 - 1).unsqueeze(1)
        point_bev_features = F.grid_sample(
            bev_features, grid=grid_sample_xy, align_corners=True)
        return point_bev_features.squeeze(2).permute(0, 2, 1).contiguous()

    def aggregate_keypoint_features(
            self,
            aggregate_module: nn.Module,
            points_xyz: Tensor,
            features: Optional[Tensor] = None,
            target_xyz: Optional[Tensor] = None) -> Tuple:
        """Aggregate keypoint features from one source by aggregate module.

        Args:
            aggregate_module (nn.Module): A SA module is used to aggregate
                features.
            points_xyz (torch.Tensor): (B, N, 3) xyz coordinates of the
                features.
            features (torch.Tensor, optional): (B, C, N) features of each
                point. Default: None.
            target_xyz (torch.Tensor, optional): (B, M, 3) new coords of
                the outputs. Default: None.

        Returns:
            Tensor: New features xyz. (B, M, 3) where M is the number of
                points.
            Tensor: New feature descriptors.(B, M, C) where M is the number
                of points. C is the sum of multi-scale SA modules output
                channels num, like sum_k(mlps[k][-1]).
            Tensor: Index of the features. (B, M) where M is the number
                of points.
        """
        new_xyz, pooled_features, indices = aggregate_module(
            points_xyz=points_xyz.contiguous(),
            features=features.contiguous(),
            target_xyz=target_xyz.contiguous())
        return new_xyz, pooled_features.transpose(1, 2), indices

    def sample_key_points(self, batch_inputs_dict: dict) -> Tensor:
        """Sample key points from raw input points.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxel_dict' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - voxel_dict (dict , optional): Voxel dict of each sample.

        Returns:
            torch.Tensor: Batch key points, shape is (B,N,D),
                where N is key points num, D is raw points dim.
        """
        if self.sample_model == 'voxel_centers':
            coors = batch_inputs_dict['voxel_dict']['coors']
            batch_key_points, _ = self.get_voxel_centers(coors, scale_factor=1)
        else:
            raw_points = self.padded_batch_points(batch_inputs_dict['points'])
            points_xyz = raw_points[:, :, :3].contiguous()
            points_flipped = raw_points.transpose(1, 2).contiguous()
            indices = self.keypoints_sampler(points_xyz, None)
            batch_key_points = gather_points(
                points_flipped, indices).transpose(1, 2).contiguous()
        return batch_key_points

    def padded_batch_points(self, points_list: List[Tensor]) -> Tensor:
        """Padded points list to a Tensor.

        Args:
            points_list (List[torch.Tensor]): Point cloud of each sample.

        Returns:
            torch.Tensor: Batch points tensor, shape is (B,N,D),
                where N is points num,D is raw points dim.
        """
        padded_points_list = []
        max_points_num = max(points.shape[0] for points in points_list)
        for points in points_list:
            padded_points_list.append(
                F.pad(points, (0, 0, 0, max_points_num - points.shape[0])))
        batch_points = torch.stack(padded_points_list, dim=0)
        return batch_points

    def get_voxel_centers(self, coors: Tensor, scale_factor: int = 1) -> Tuple:
        """Get voxel centers.

        Args:
            coors (torch.Tensor): The coordinates of each voxel.
            scale_factor (int): Down sample scale factor.

        Returns:
            Tuple[torch.Tensor, list]: Return batch voxel centers and
                voxel num.
        """
        assert coors.shape[1] == 4
        batch_size = coors[-1][0] + 1
        batch_centers = []
        batch_voxel_num = []
        voxel_centers = coors[:, [3, 2, 1]].float()
        voxel_size = voxel_centers.new_tensor(self.voxel_size)
        pc_range_min = voxel_centers.new_tensor(self.point_cloud_range[:3])

        voxel_centers = voxel_centers * voxel_size * scale_factor
        voxel_centers += pc_range_min
        voxel_centers.add_(0.5 * voxel_size * scale_factor)
        for i in range(batch_size):
            cur_voxel_centers = voxel_centers[coors[:, 0] == i]
            batch_centers.append(cur_voxel_centers)
            batch_voxel_num.append(cur_voxel_centers.shape[0])
        batch_centers = self.padded_batch_points(batch_centers)
        return batch_centers, batch_voxel_num

    def forward(self, batch_inputs_dict: dict, feats_dict: dict,
                rpn_results_list: List[InstanceData]) -> dict:
        """Extract key points features from multi-source features.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.
            feats_dict (dict): Multi-source features.
            rpn_results_list (List[:obj:`InstancesData`], optional): Detection
                results of rpn head.

        Returns:
            Dict: Include key points and key points features.
        """
        batch_inputs_dict['voxel_dict'] = feats_dict.pop('voxel_dict')
        batch_keypoints = self.sample_key_points(batch_inputs_dict)[..., :3]
        batch_raw_points = self.padded_batch_points(
            batch_inputs_dict['points'])
        point_features_list = []
        num_keypoints = batch_keypoints.shape[1]

        if self.bev_cfg:
            spatial_feats = feats_dict['spatial_feats']
            keypoint_bev_features = self.interpolate_from_bev_features(
                batch_keypoints, spatial_feats, self.bev_cfg.scale_factor)
            point_features_list.append(keypoint_bev_features)

        if self.rawpoint_sa_cfg:
            batch_raw_points_feats = batch_raw_points[:, :, 3:].transpose(1, 2)
            _, pooled_features, _ = \
                self.aggregate_keypoint_features(self.rawpoints_sa_layer,
                                                 batch_raw_points[:, :, :3],
                                                 batch_raw_points_feats,
                                                 batch_keypoints)
            point_features_list.append(pooled_features)

        for i, voxel_sa_layer in enumerate(self.voxel_sa_layers):
            cur_coords = feats_dict['multi_scale_3d_feats'][i].indices
            cur_feats = feats_dict['multi_scale_3d_feats'][
                i].features.contiguous()
            batch_voxel_centers, batch_voxel_num = self.get_voxel_centers(
                coors=cur_coords, scale_factor=self.downsample_scale_factor[i])
            batch_cur_feats = self.padded_batch_points(
                torch.split(cur_feats, batch_voxel_num, dim=0))
            _, pooled_features, _ = self.aggregate_keypoint_features(
                self.voxel_sa_layers[i],
                points_xyz=batch_voxel_centers.contiguous(),
                features=batch_cur_feats.transpose(1, 2).contiguous(),
                target_xyz=batch_keypoints[:, :, :3].contiguous())
            point_features_list.append(pooled_features)
        batch_size = len(batch_inputs_dict['points'])
        keypoint_features = torch.cat(
            point_features_list, dim=-1).view(batch_size * num_keypoints, -1)
        fusion_point_features = self.point_feature_fusion(keypoint_features)

        return dict(
            keypoint_features=keypoint_features,
            fusion_keypoint_features=fusion_point_features,
            keypoints=batch_keypoints)
