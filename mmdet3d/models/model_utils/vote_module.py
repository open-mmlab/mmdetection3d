import torch
from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet3d.models.builder import build_loss


class VoteModule(nn.Module):
    """Vote module.

    Generate votes from seed point features.

    Args:
        in_channels (int): Number of channels of seed point features.
        vote_per_seed (int): Number of votes generated from each seed point.
        gt_per_seed (int): Number of ground truth votes generated
            from each seed point.
        conv_channels (tuple[int]): Out channels of vote
            generating convolution.
        conv_cfg (dict): Config of convolution.
            Default: dict(type='Conv1d').
        norm_cfg (dict): Config of normalization.
            Default: dict(type='BN1d').
        norm_feats (bool): Whether to normalize features.
            Default: True.
        vote_loss (dict): Config of vote loss.
    """

    def __init__(self,
                 in_channels,
                 vote_per_seed=1,
                 gt_per_seed=3,
                 conv_channels=(16, 16),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 norm_feats=True,
                 vote_loss=None):
        super().__init__()
        self.in_channels = in_channels
        self.vote_per_seed = vote_per_seed
        self.gt_per_seed = gt_per_seed
        self.norm_feats = norm_feats
        self.vote_loss = build_loss(vote_loss)

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
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[k]
        self.vote_conv = nn.Sequential(*vote_conv_list)

        # conv_out predicts coordinate and residual features
        out_channel = (3 + in_channels) * self.vote_per_seed
        self.conv_out = nn.Conv1d(prev_channels, out_channel, 1)

    def forward(self, seed_points, seed_feats):
        """forward.

        Args:
            seed_points (torch.Tensor): Coordinate of the seed
                points in shape (B, N, 3).
            seed_feats (torch.Tensor): Features of the seed points in shape
                (B, C, N).

        Returns:
            tuple[torch.Tensor]:

                - vote_points: Voted xyz based on the seed points \
                    with shape (B, M, 3), ``M=num_seed*vote_per_seed``.
                - vote_features: Voted features based on the seed points with \
                    shape (B, C, M) where ``M=num_seed*vote_per_seed``, \
                    ``C=vote_feature_dim``.
        """
        batch_size, feat_channels, num_seed = seed_feats.shape
        num_vote = num_seed * self.vote_per_seed
        x = self.vote_conv(seed_feats)
        # (batch_size, (3+out_dim)*vote_per_seed, num_seed)
        votes = self.conv_out(x)

        votes = votes.transpose(2, 1).view(batch_size, num_seed,
                                           self.vote_per_seed, -1)
        offset = votes[:, :, :, 0:3].contiguous()
        res_feats = votes[:, :, :, 3:].contiguous()

        vote_points = seed_points.unsqueeze(2) + offset
        vote_points = vote_points.view(batch_size, num_vote, 3)
        vote_feats = seed_feats.permute(
            0, 2, 1).unsqueeze(2).contiguous() + res_feats
        vote_feats = vote_feats.view(batch_size, num_vote,
                                     feat_channels).transpose(2,
                                                              1).contiguous()

        if self.norm_feats:
            features_norm = torch.norm(vote_feats, p=2, dim=1)
            vote_feats = vote_feats.div(features_norm.unsqueeze(1))
        return vote_points, vote_feats

    def get_loss(self, seed_points, vote_points, seed_indices,
                 vote_targets_mask, vote_targets):
        """Calculate loss of voting module.

        Args:
            seed_points (torch.Tensor): Coordinate of the seed points.
            vote_points (torch.Tensor): Coordinate of the vote points.
            seed_indices (torch.Tensor): Indices of seed points in raw points.
            vote_targets_mask (torch.Tensor): Mask of valid vote targets.
            vote_targets (torch.Tensor): Targets of votes.

        Returns:
            torch.Tensor: Weighted vote loss.
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
