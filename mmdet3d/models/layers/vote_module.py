# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.cnn import ConvModule
from mmengine import is_tuple_of
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType


class VoteModule(nn.Module):
    """Vote module.

    Generate votes from seed point features.

    Args:
        in_channels (int): Number of channels of seed point features.
        vote_per_seed (int): Number of votes generated from each seed point.
            Defaults to 1.
        gt_per_seed (int): Number of ground truth votes generated from each
            seed point. Defaults to 3.
        num_points (int): Number of points to be used for voting.
            Defaults to 1.
        conv_channels (tuple[int]): Out channels of vote generating
            convolution. Defaults to (16, 16).
        conv_cfg (:obj:`ConfigDict` or dict): Config dict for convolution
            layer. Defaults to dict(type='Conv1d').
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d').
        norm_feats (bool): Whether to normalize features. Default to True.
        with_res_feat (bool): Whether to predict residual features.
            Defaults to True.
        vote_xyz_range (List[float], optional): The range of points
            translation. Defaults to None.
        vote_loss (:obj:`ConfigDict` or dict, optional): Config of vote loss.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 vote_per_seed: int = 1,
                 gt_per_seed: int = 3,
                 num_points: int = -1,
                 conv_channels: Tuple[int] = (16, 16),
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 norm_feats: bool = True,
                 with_res_feat: bool = True,
                 vote_xyz_range: List[float] = None,
                 vote_loss: OptConfigType = None) -> None:
        super(VoteModule, self).__init__()
        self.in_channels = in_channels
        self.vote_per_seed = vote_per_seed
        self.gt_per_seed = gt_per_seed
        self.num_points = num_points
        self.norm_feats = norm_feats
        self.with_res_feat = with_res_feat

        assert vote_xyz_range is None or is_tuple_of(vote_xyz_range, float)
        self.vote_xyz_range = vote_xyz_range

        if vote_loss is not None:
            self.vote_loss = MODELS.build(vote_loss)

        prev_channels = in_channels
        vote_conv_list = list()
        for k in range(len(conv_channels)):
            vote_conv_list.append(
                ConvModule(
                    prev_channels,
                    conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[k]
        self.vote_conv = nn.Sequential(*vote_conv_list)

        # conv_out predicts coordinate and residual features
        if with_res_feat:
            out_channel = (3 + in_channels) * self.vote_per_seed
        else:
            out_channel = 3 * self.vote_per_seed
        self.conv_out = nn.Conv1d(prev_channels, out_channel, 1)

    def forward(self, seed_points: Tensor,
                seed_feats: Tensor) -> Tuple[Tensor]:
        """Forward.

        Args:
            seed_points (Tensor): Coordinate of the seed points in shape
                (B, N, 3).
            seed_feats (Tensor): Features of the seed points in shape
                (B, C, N).

        Returns:
            Tuple[torch.Tensor]:

                - vote_points: Voted xyz based on the seed points
                  with shape (B, M, 3), ``M=num_seed*vote_per_seed``.
                - vote_features: Voted features based on the seed points with
                  shape (B, C, M) where ``M=num_seed*vote_per_seed``,
                  ``C=vote_feature_dim``.
        """
        if self.num_points != -1:
            assert self.num_points < seed_points.shape[1], \
                f'Number of vote points ({self.num_points}) should be '\
                f'smaller than seed points size ({seed_points.shape[1]})'
            seed_points = seed_points[:, :self.num_points]
            seed_feats = seed_feats[..., :self.num_points]

        batch_size, feat_channels, num_seed = seed_feats.shape
        num_vote = num_seed * self.vote_per_seed
        x = self.vote_conv(seed_feats)
        # (batch_size, (3+out_dim)*vote_per_seed, num_seed)
        votes = self.conv_out(x)

        votes = votes.transpose(2, 1).view(batch_size, num_seed,
                                           self.vote_per_seed, -1)

        offset = votes[:, :, :, 0:3]
        if self.vote_xyz_range is not None:
            limited_offset_list = []
            for axis in range(len(self.vote_xyz_range)):
                limited_offset_list.append(offset[..., axis].clamp(
                    min=-self.vote_xyz_range[axis],
                    max=self.vote_xyz_range[axis]))
            limited_offset = torch.stack(limited_offset_list, -1)
            vote_points = (seed_points.unsqueeze(2) +
                           limited_offset).contiguous()
        else:
            vote_points = (seed_points.unsqueeze(2) + offset).contiguous()
        vote_points = vote_points.view(batch_size, num_vote, 3)
        offset = offset.reshape(batch_size, num_vote, 3).transpose(2, 1)

        if self.with_res_feat:
            res_feats = votes[:, :, :, 3:]
            vote_feats = (seed_feats.transpose(2, 1).unsqueeze(2) +
                          res_feats).contiguous()
            vote_feats = vote_feats.view(batch_size,
                                         num_vote, feat_channels).transpose(
                                             2, 1).contiguous()

            if self.norm_feats:
                features_norm = torch.norm(vote_feats, p=2, dim=1)
                vote_feats = vote_feats.div(features_norm.unsqueeze(1))
        else:
            vote_feats = seed_feats
        return vote_points, vote_feats, offset

    def get_loss(self, seed_points: Tensor, vote_points: Tensor,
                 seed_indices: Tensor, vote_targets_mask: Tensor,
                 vote_targets: Tensor) -> Tensor:
        """Calculate loss of voting module.

        Args:
            seed_points (Tensor): Coordinate of the seed points.
            vote_points (Tensor): Coordinate of the vote points.
            seed_indices (Tensor): Indices of seed points in raw points.
            vote_targets_mask (Tensor): Mask of valid vote targets.
            vote_targets (Tensor): Targets of votes.

        Returns:
            Tensor: Weighted vote loss.
        """
        batch_size, num_seed = seed_points.shape[:2]

        seed_gt_votes_mask = torch.gather(vote_targets_mask, 1,
                                          seed_indices).float()

        seed_indices_expand = seed_indices.unsqueeze(-1).repeat(
            1, 1, 3 * self.gt_per_seed)
        seed_gt_votes = torch.gather(vote_targets, 1, seed_indices_expand)
        seed_gt_votes += seed_points.repeat(1, 1, self.gt_per_seed)

        weight = seed_gt_votes_mask / (torch.sum(seed_gt_votes_mask) + 1e-6)
        distance = self.vote_loss(
            vote_points.view(batch_size * num_seed, -1, 3),
            seed_gt_votes.view(batch_size * num_seed, -1, 3),
            dst_weight=weight.view(batch_size * num_seed, 1))[1]
        vote_loss = torch.sum(torch.min(distance, dim=1)[0])

        return vote_loss
