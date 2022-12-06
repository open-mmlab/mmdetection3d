# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List

import numpy as np
import torch
from mmcv.ops.furthest_point_sample import furthest_point_sample
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class FPSSampler(nn.Module):
    """Using Euclidean distances of points for FPS.

    Args:
        num_keypoints (int): Need to sample key points num.
    """

    def __init__(self, num_keypoints: int) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

    def forward(self, points_list: List[Tensor], **kwargs) -> Tensor:
        """Sampling points with D-FPS."""
        sampled_points = []
        for batch_idx in range(len(points_list)):
            points = points_list[batch_idx]
            num_points = points.shape[0]
            fps_idx = furthest_point_sample(
                points.unsqueeze(dim=0).contiguous(),
                self.num_keypoints).long()[0]
            if num_points < self.num_keypoints:
                times = int(self.num_keypoints / num_points) + 1
                non_empty = fps_idx[:num_points]
                fps_idx = non_empty.repeat(times)[:self.num_keypoints]
            key_points = points[fps_idx]
            sampled_points.append(key_points)
        return sampled_points


@MODELS.register_module()
class SPCSampler(nn.Module):
    """Using Sectorized Proposal-Centric Sampling for Efficient and
    Representative Keypoint Sampling.

    Args:
        num_keypoints (int): Need to sample key points num.
        sample_radius_with_roi (float): Sample points radius of each roi boxes.
        num_sectors (int): Divide space into `num_sectors` sectors.
        num_max_points_of_part (int): Max points num in each part.
            Default to 200000.
    """

    def __init__(
        self,
        num_keypoints: int,
        sample_radius_with_roi: float,
        num_sectors: int,
        num_max_points_of_part: int = 200000,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sample_radius_with_roi = sample_radius_with_roi
        self.num_max_points_of_part = num_max_points_of_part
        self.num_sectors = num_sectors

    def sample_points_with_roi(self, rois: Tensor, points: Tensor) -> Tensor:
        """Sample points with roi boxes. Filter some points which keep away roi
        boxes.

        Args:
            rois (torch.Tensor): (M, 7 + C) Roi boxes.
            points (torch.Tensor): (N, 3) Input raw points coordinates.

        Returns:
            torch.Tensor: (N_out, 3) Sampled points.
        """
        if points.shape[0] < self.num_max_points_of_part:
            distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            point_mask = min_dis < roi_max_dim + self.sample_radius_with_roi
        else:
            start_idx = 0
            point_mask_list = []
            while start_idx < points.shape[0]:
                distance = (points[start_idx:start_idx +
                                   self.num_max_points_of_part, None, :] -
                            rois[None, :, 0:3]).norm(dim=-1)
                min_dis, min_dis_roi_idx = distance.min(dim=-1)
                roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
                cur_point_mask = \
                    min_dis < roi_max_dim + self.sample_radius_with_roi
                point_mask_list.append(cur_point_mask)
                start_idx += self.num_max_points_of_part
            point_mask = torch.cat(point_mask_list, dim=0)

        sampled_points = points[:1] if point_mask.sum() == 0 else points[
            point_mask, :]

        return sampled_points, point_mask

    def sector_fps(self, points: Tensor) -> Tensor:
        """Use FPS sample points in each sector.

        Args:
            points (torch.tensor): Input points coordinates.

        Returns:
            torch.tensor: (N_out, 3) Sampled points.
        """
        sector_size = np.pi * 2 / self.num_sectors
        point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
        sector_idx = (point_angles / sector_size).floor().clamp(
            min=0, max=self.num_sectors)
        xyz_points_list = []
        xyz_batch_cnt = []
        num_sampled_points_list = []
        for k in range(self.num_sectors):
            mask = (sector_idx == k)
            cur_num_points = mask.sum().item()
            if cur_num_points > 0:
                xyz_points_list.append(points[mask])
                xyz_batch_cnt.append(cur_num_points)
                ratio = cur_num_points / points.shape[0]
                num_sampled_points_list.append(
                    min(cur_num_points, math.ceil(ratio * self.num_keypoints)))

        if len(xyz_batch_cnt) == 0:
            xyz_points_list.append(points)
            xyz_batch_cnt.append(len(points))
            num_sampled_points_list.append(self.num_keypoints)
            print(f'Warning: empty sector points detected in SectorFPS: '
                  f'points.shape={points.shape}')

        xyz = torch.cat(xyz_points_list, dim=0)
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()

        sampled_pt_idxs = furthest_point_sample(xyz.contiguous(),
                                                num_sampled_points_list,
                                                xyz_batch_cnt).long()

        sampled_points = xyz[sampled_pt_idxs]

        return sampled_points

    def sectorized_proposal_centric_sampling(self, roi_boxes: Tensor,
                                             points: Tensor) -> Tensor:
        """Sampled key points by roi and sector fps.

        Args:
            roi_boxes (torch.Tensor): Roi boxes used to sample points.
            points (torch.Tensor): Input points.

        Returns:
            torch.Tensor: Sampled points.
        """
        sampled_points, _ = self.sample_points_with_roi(
            rois=roi_boxes, points=points)
        sampled_points = self.sector_fps(points=sampled_points)
        return sampled_points

    def forward(self, points_list: List[Tensor],
                roi_boxes_list) -> List[Tensor]:
        """Sampling points with SPC.

        Args:
            points_list (List[torch.Tensor]): Input batch points list.
            roi_boxes_list (List[:obj:`InstanceData`]): A list include
                some roi boxes.

        Returns:
            List[torch.Tensor]: Sampled points results.
        """
        key_points_list = []
        for i in range(len(roi_boxes_list)):
            boxes = roi_boxes_list[i]
            gemo_center_boxes = boxes.bboxes_3d.tensor.clone()
            gemo_center_boxes[:, 2] = \
                gemo_center_boxes.bboxes_3d.tensor[:, 2] + \
                gemo_center_boxes.bboxes_3d.tensor[:, 5] / 2
            roi_boxes_list[i] = gemo_center_boxes
        for batch_idx in range(len(points_list)):
            points = points_list[batch_idx]
            cur_keypoints = self.sectorized_proposal_centric_sampling(
                roi_boxes=roi_boxes_list[batch_idx], points=points)
            num_points = cur_keypoints.shape[0]
            if num_points < self.num_keypoints:
                times = int(self.num_keypoints / num_points) + 1
                sampled_keypoinys = cur_keypoints.repeat(
                    times, 1)[:self.num_keypoints]
            else:
                sampled_keypoinys = cur_keypoints[:self.num_keypoints]

            key_points_list.append(sampled_keypoinys)
        return key_points_list
