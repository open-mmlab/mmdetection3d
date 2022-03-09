# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmcv.runner import force_fp32
from torch import nn as nn

from .furthest_point_sample import (furthest_point_sample,
                                    furthest_point_sample_with_dist)
from .utils import calc_square_dist


def get_sampler_type(sampler_type):
    """Get the type and mode of points sampler.

    Args:
        sampler_type (str): The type of points sampler.
            The valid value are "D-FPS", "F-FPS", or "FS".

    Returns:
        class: Points sampler type.
    """
    if sampler_type == 'D-FPS':
        sampler = DFPS_Sampler
    elif sampler_type == 'F-FPS':
        sampler = FFPS_Sampler
    elif sampler_type == 'FS':
        sampler = FS_Sampler
    else:
        raise ValueError('Only "sampler_type" of "D-FPS", "F-FPS", or "FS"'
                         f' are supported, got {sampler_type}')

    return sampler


class Points_Sampler(nn.Module):
    """Points sampling.

    Args:
        num_point (list[int]): Number of sample points.
        fps_mod_list (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional):
            Range of points to apply FPS. Default: [-1].
    """

    def __init__(self,
                 num_point: List[int],
                 fps_mod_list: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1]):
        super(Points_Sampler, self).__init__()
        # FPS would be applied to different fps_mod in the list,
        # so the length of the num_point should be equal to
        # fps_mod_list and fps_sample_range_list.
        assert len(num_point) == len(fps_mod_list) == len(
            fps_sample_range_list)
        self.num_point = num_point
        self.fps_sample_range_list = fps_sample_range_list
        self.samplers = nn.ModuleList()
        for fps_mod in fps_mod_list:
            self.samplers.append(get_sampler_type(fps_mod)())
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, points_xyz, features):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) Descriptors of the features.

        Return：
            Tensor: (B, npoint, sample_num) Indices of sampled points.
        """
        indices = []
        last_fps_end_index = 0

        for fps_sample_range, sampler, npoint in zip(
                self.fps_sample_range_list, self.samplers, self.num_point):
            assert fps_sample_range < points_xyz.shape[1]

            if fps_sample_range == -1:
                sample_points_xyz = points_xyz[:, last_fps_end_index:]
                sample_features = features[:, :, last_fps_end_index:] if \
                    features is not None else None
            else:
                sample_points_xyz = \
                    points_xyz[:, last_fps_end_index:fps_sample_range]
                sample_features = \
                    features[:, :, last_fps_end_index:fps_sample_range] if \
                    features is not None else None

            fps_idx = sampler(sample_points_xyz.contiguous(), sample_features,
                              npoint)

            indices.append(fps_idx + last_fps_end_index)
            last_fps_end_index += fps_sample_range
        indices = torch.cat(indices, dim=1)

        return indices


class DFPS_Sampler(nn.Module):
    """DFPS_Sampling.

    Using Euclidean distances of points for FPS.
    """

    def __init__(self):
        super(DFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with D-FPS."""
        fps_idx = furthest_point_sample(points.contiguous(), npoint)
        return fps_idx


class FFPS_Sampler(nn.Module):
    """FFPS_Sampler.

    Using feature distances for FPS.
    """

    def __init__(self):
        super(FFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with F-FPS."""
        assert features is not None, \
            'feature input to FFPS_Sampler should not be None'
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(
            features_for_fps, features_for_fps, norm=False)
        fps_idx = furthest_point_sample_with_dist(features_dist, npoint)
        return fps_idx


class FS_Sampler(nn.Module):
    """FS_Sampling.

    Using F-FPS and D-FPS simultaneously.
    """

    def __init__(self):
        super(FS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with FS_Sampling."""
        assert features is not None, \
            'feature input to FS_Sampler should not be None'
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(
            features_for_fps, features_for_fps, norm=False)
        fps_idx_ffps = furthest_point_sample_with_dist(features_dist, npoint)
        fps_idx_dfps = furthest_point_sample(points, npoint)
        fps_idx = torch.cat([fps_idx_ffps, fps_idx_dfps], dim=1)
        return fps_idx
