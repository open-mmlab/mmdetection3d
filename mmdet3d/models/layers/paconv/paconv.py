# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcv.cnn import (ConvModule, build_activation_layer, build_norm_layer,
                      constant_init)
from mmcv.ops import assign_score_withk as assign_score_cuda
from torch import nn as nn
from torch.nn import functional as F

from .utils import assign_kernel_withoutk, assign_score, calc_euclidian_dist


class ScoreNet(nn.Module):
    r"""ScoreNet that outputs coefficient scores to assemble kernel weights in
    the weight bank according to the relative position of point pairs.

    Args:
        mlp_channels (List[int]): Hidden unit sizes of SharedMLP layers.
        last_bn (bool, optional): Whether to use BN on the last output of mlps.
            Defaults to False.
        score_norm (str, optional): Normalization function of output scores.
            Can be 'softmax', 'sigmoid' or 'identity'. Defaults to 'softmax'.
        temp_factor (float, optional): Temperature factor to scale the output
            scores before softmax. Defaults to 1.0.
        norm_cfg (dict, optional): Type of normalization method.
            Defaults to dict(type='BN2d').
        bias (bool | str, optional): If specified as `auto`, it will be decided
            by the norm_cfg. Bias will be set as True if `norm_cfg` is None,
            otherwise False. Defaults to 'auto'.

    Note:
        The official code applies xavier_init to all Conv layers in ScoreNet,
            see `PAConv <https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg
            /model/pointnet2/paconv.py#L105>`_. However in our experiments, we
            did not find much difference in applying such xavier initialization
            or not. So we neglect this initialization in our implementation.
    """

    def __init__(self,
                 mlp_channels,
                 last_bn=False,
                 score_norm='softmax',
                 temp_factor=1.0,
                 norm_cfg=dict(type='BN2d'),
                 bias='auto'):
        super(ScoreNet, self).__init__()

        assert score_norm in ['softmax', 'sigmoid', 'identity'], \
            f'unsupported score_norm function {score_norm}'

        self.score_norm = score_norm
        self.temp_factor = temp_factor

        self.mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 2):
            self.mlps.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg,
                    bias=bias))

        # for the last mlp that outputs scores, no relu and possibly no bn
        i = len(mlp_channels) - 2
        self.mlps.add_module(
            f'layer{i}',
            ConvModule(
                mlp_channels[i],
                mlp_channels[i + 1],
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg if last_bn else None,
                act_cfg=None,
                bias=bias))

    def forward(self, xyz_features):
        """Forward.

        Args:
            xyz_features (torch.Tensor): (B, C, N, K), features constructed
                from xyz coordinates of point pairs. May contain relative
                positions, Euclidean distance, etc.

        Returns:
            torch.Tensor: (B, N, K, M), predicted scores for `M` kernels.
        """
        scores = self.mlps(xyz_features)  # (B, M, N, K)

        # perform score normalization
        if self.score_norm == 'softmax':
            scores = F.softmax(scores / self.temp_factor, dim=1)
        elif self.score_norm == 'sigmoid':
            scores = torch.sigmoid(scores / self.temp_factor)
        else:  # 'identity'
            scores = scores

        scores = scores.permute(0, 2, 3, 1)  # (B, N, K, M)

        return scores


class PAConv(nn.Module):
    """Non-CUDA version of PAConv.

    PAConv stores a trainable weight bank containing several kernel weights.
    Given input points and features, it computes coefficient scores to assemble
    those kernels to form conv kernels, and then runs convolution on the input.

    Args:
        in_channels (int): Input channels of point features.
        out_channels (int): Output channels of point features.
        num_kernels (int): Number of kernel weights in the weight bank.
        norm_cfg (dict, optional): Type of normalization method.
            Defaults to dict(type='BN2d', momentum=0.1).
        act_cfg (dict, optional): Type of activation method.
            Defaults to dict(type='ReLU', inplace=True).
        scorenet_input (str, optional): Type of input to ScoreNet.
            Can be 'identity', 'w_neighbor' or 'w_neighbor_dist'.
            Defaults to 'w_neighbor_dist'.
        weight_bank_init (str, optional): Init method of weight bank kernels.
            Can be 'kaiming' or 'xavier'. Defaults to 'kaiming'.
        kernel_input (str, optional): Input features to be multiplied with
            kernel weights. Can be 'identity' or 'w_neighbor'.
            Defaults to 'w_neighbor'.
        scorenet_cfg (dict, optional): Config of the ScoreNet module, which
            may contain the following keys and values:

            - mlp_channels (List[int]): Hidden units of MLPs.
            - score_norm (str): Normalization function of output scores.
                Can be 'softmax', 'sigmoid' or 'identity'.
            - temp_factor (float): Temperature factor to scale the output
                scores before softmax.
            - last_bn (bool): Whether to use BN on the last output of mlps.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 norm_cfg=dict(type='BN2d', momentum=0.1),
                 act_cfg=dict(type='ReLU', inplace=True),
                 scorenet_input='w_neighbor_dist',
                 weight_bank_init='kaiming',
                 kernel_input='w_neighbor',
                 scorenet_cfg=dict(
                     mlp_channels=[16, 16, 16],
                     score_norm='softmax',
                     temp_factor=1.0,
                     last_bn=False)):
        super(PAConv, self).__init__()

        # determine weight kernel size according to used features
        if kernel_input == 'identity':
            # only use grouped_features
            kernel_mul = 1
        elif kernel_input == 'w_neighbor':
            # concat of (grouped_features - center_features, grouped_features)
            kernel_mul = 2
        else:
            raise NotImplementedError(
                f'unsupported kernel_input {kernel_input}')
        self.kernel_input = kernel_input
        in_channels = kernel_mul * in_channels

        # determine mlp channels in ScoreNet according to used xyz features
        if scorenet_input == 'identity':
            # only use relative position (grouped_xyz - center_xyz)
            self.scorenet_in_channels = 3
        elif scorenet_input == 'w_neighbor':
            # (grouped_xyz - center_xyz, grouped_xyz)
            self.scorenet_in_channels = 6
        elif scorenet_input == 'w_neighbor_dist':
            # (center_xyz, grouped_xyz - center_xyz, Euclidean distance)
            self.scorenet_in_channels = 7
        else:
            raise NotImplementedError(
                f'unsupported scorenet_input {scorenet_input}')
        self.scorenet_input = scorenet_input

        # construct kernel weights in weight bank
        # self.weight_bank is of shape [C, num_kernels * out_c]
        # where C can be in_c or (2 * in_c)
        if weight_bank_init == 'kaiming':
            weight_init = nn.init.kaiming_normal_
        elif weight_bank_init == 'xavier':
            weight_init = nn.init.xavier_normal_
        else:
            raise NotImplementedError(
                f'unsupported weight bank init method {weight_bank_init}')

        self.num_kernels = num_kernels  # the parameter `m` in the paper
        weight_bank = weight_init(
            torch.empty(self.num_kernels, in_channels, out_channels))
        weight_bank = weight_bank.permute(1, 0, 2).reshape(
            in_channels, self.num_kernels * out_channels).contiguous()
        self.weight_bank = nn.Parameter(weight_bank, requires_grad=True)

        # construct ScoreNet
        scorenet_cfg_ = copy.deepcopy(scorenet_cfg)
        scorenet_cfg_['mlp_channels'].insert(0, self.scorenet_in_channels)
        scorenet_cfg_['mlp_channels'].append(self.num_kernels)
        self.scorenet = ScoreNet(**scorenet_cfg_)

        self.bn = build_norm_layer(norm_cfg, out_channels)[1] if \
            norm_cfg is not None else None
        self.activate = build_activation_layer(act_cfg) if \
            act_cfg is not None else None

        # set some basic attributes of Conv layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.init_weights()

    def init_weights(self):
        """Initialize weights of shared MLP layers and BN layers."""
        if self.bn is not None:
            constant_init(self.bn, val=1, bias=0)

    def _prepare_scorenet_input(self, points_xyz):
        """Prepare input point pairs features for self.ScoreNet.

        Args:
            points_xyz (torch.Tensor): (B, 3, npoint, K)
                Coordinates of the grouped points.

        Returns:
            torch.Tensor: (B, C, npoint, K)
                The generated features per point pair.
        """
        B, _, npoint, K = points_xyz.size()
        center_xyz = points_xyz[..., :1].repeat(1, 1, 1, K)
        xyz_diff = points_xyz - center_xyz  # [B, 3, npoint, K]
        if self.scorenet_input == 'identity':
            xyz_features = xyz_diff
        elif self.scorenet_input == 'w_neighbor':
            xyz_features = torch.cat((xyz_diff, points_xyz), dim=1)
        else:  # w_neighbor_dist
            euclidian_dist = calc_euclidian_dist(
                center_xyz.permute(0, 2, 3, 1).reshape(B * npoint * K, 3),
                points_xyz.permute(0, 2, 3, 1).reshape(B * npoint * K, 3)).\
                    reshape(B, 1, npoint, K)
            xyz_features = torch.cat((center_xyz, xyz_diff, euclidian_dist),
                                     dim=1)
        return xyz_features

    def forward(self, inputs):
        """Forward.

        Args:
            inputs (tuple(torch.Tensor)):

                - features (torch.Tensor): (B, in_c, npoint, K)
                    Features of the queried points.
                - points_xyz (torch.Tensor): (B, 3, npoint, K)
                    Coordinates of the grouped points.

        Returns:
            Tuple[torch.Tensor]:

                - new_features: (B, out_c, npoint, K), features after PAConv.
                - points_xyz: same as input.
        """
        features, points_xyz = inputs
        B, _, npoint, K = features.size()

        if self.kernel_input == 'w_neighbor':
            center_features = features[..., :1].repeat(1, 1, 1, K)
            features_diff = features - center_features
            # to (B, 2 * in_c, npoint, K)
            features = torch.cat((features_diff, features), dim=1)

        # prepare features for between each point and its grouping center
        xyz_features = self._prepare_scorenet_input(points_xyz)

        # scores to assemble kernel weights
        scores = self.scorenet(xyz_features)  # [B, npoint, K, m]

        # first compute out features over all kernels
        # features is [B, C, npoint, K], weight_bank is [C, m * out_c]
        new_features = torch.matmul(
            features.permute(0, 2, 3, 1),
            self.weight_bank).view(B, npoint, K, self.num_kernels,
                                   -1)  # [B, npoint, K, m, out_c]

        # then aggregate using scores
        new_features = assign_score(scores, new_features)
        # to [B, out_c, npoint, K]
        new_features = new_features.permute(0, 3, 1, 2).contiguous()

        if self.bn is not None:
            new_features = self.bn(new_features)
        if self.activate is not None:
            new_features = self.activate(new_features)

        # in order to keep input output consistency
        # so that we can wrap PAConv in Sequential
        return (new_features, points_xyz)


class PAConvCUDA(PAConv):
    """CUDA version of PAConv that implements a cuda op to efficiently perform
    kernel assembling.

    Different from vanilla PAConv, the input features of this function is not
    grouped by centers. Instead, they will be queried on-the-fly by the
    additional input `points_idx`. This avoids the large intermediate matrix.
    See the `paper <https://arxiv.org/pdf/2103.14635.pdf>`_ appendix Sec. D for
    more detailed descriptions.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 norm_cfg=dict(type='BN2d', momentum=0.1),
                 act_cfg=dict(type='ReLU', inplace=True),
                 scorenet_input='w_neighbor_dist',
                 weight_bank_init='kaiming',
                 kernel_input='w_neighbor',
                 scorenet_cfg=dict(
                     mlp_channels=[8, 16, 16],
                     score_norm='softmax',
                     temp_factor=1.0,
                     last_bn=False)):
        super(PAConvCUDA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_kernels=num_kernels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            scorenet_input=scorenet_input,
            weight_bank_init=weight_bank_init,
            kernel_input=kernel_input,
            scorenet_cfg=scorenet_cfg)

        assert self.kernel_input == 'w_neighbor', \
            'CUDA implemented PAConv only supports w_neighbor kernel_input'

    def forward(self, inputs):
        """Forward.

        Args:
            inputs (tuple(torch.Tensor)):

                - features (torch.Tensor): (B, in_c, N)
                    Features of all points in the current point cloud.
                    Different from non-CUDA version PAConv, here the features
                        are not grouped by each center to form a K dim.
                - points_xyz (torch.Tensor): (B, 3, npoint, K)
                    Coordinates of the grouped points.
                - points_idx (torch.Tensor): (B, npoint, K)
                    Index of the grouped points.

        Returns:
            Tuple[torch.Tensor]:

                - new_features: (B, out_c, npoint, K), features after PAConv.
                - points_xyz: same as input.
                - points_idx: same as input.
        """
        features, points_xyz, points_idx = inputs

        # prepare features for between each point and its grouping center
        xyz_features = self._prepare_scorenet_input(points_xyz)

        # scores to assemble kernel weights
        scores = self.scorenet(xyz_features)  # [B, npoint, K, m]

        # pre-compute features for points and centers separately
        # features is [B, in_c, N], weight_bank is [C, m * out_dim]
        point_feat, center_feat = assign_kernel_withoutk(
            features, self.weight_bank, self.num_kernels)

        # aggregate features using custom cuda op
        new_features = assign_score_cuda(
            scores, point_feat, center_feat, points_idx,
            'sum').contiguous()  # [B, out_c, npoint, K]

        if self.bn is not None:
            new_features = self.bn(new_features)
        if self.activate is not None:
            new_features = self.activate(new_features)

        # in order to keep input output consistency
        return (new_features, points_xyz, points_idx)
