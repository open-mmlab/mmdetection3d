# Copyright (c) OpenMMLab. All rights reserved.
import torch


def calc_square_dist(point_feat_a, point_feat_b, norm=True):
    """Calculating square distance between a and b.

    Args:
        point_feat_a (Tensor): (B, N, C) Feature vector of each point.
        point_feat_b (Tensor): (B, M, C) Feature vector of each point.
        norm (Bool, optional): Whether to normalize the distance.
            Default: True.

    Returns:
        Tensor: (B, N, M) Distance between each pair points.
    """
    length_a = point_feat_a.shape[1]
    length_b = point_feat_b.shape[1]
    num_channel = point_feat_a.shape[-1]
    # [bs, n, 1]
    a_square = torch.sum(point_feat_a.unsqueeze(dim=2).pow(2), dim=-1)
    # [bs, 1, m]
    b_square = torch.sum(point_feat_b.unsqueeze(dim=1).pow(2), dim=-1)
    a_square = a_square.repeat((1, 1, length_b))  # [bs, n, m]
    b_square = b_square.repeat((1, length_a, 1))  # [bs, n, m]

    coor = torch.matmul(point_feat_a, point_feat_b.transpose(1, 2))

    dist = a_square + b_square - 2 * coor
    if norm:
        dist = torch.sqrt(dist) / num_channel
    return dist
