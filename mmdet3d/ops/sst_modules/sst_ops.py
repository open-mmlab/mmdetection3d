# Copyright (c) OpenMMLab. All rights reserved.
import torch


def make_continuous_inds(win_inds):
    """Mapping the windows inds to continuous inds.

    The code is modified from original implementation
    https://github.com/TuSimple/SST

    Args:
        win_inds (torch.int64): Windows index of each voxel.

    Returns:
        Tensor: Continuous windows index. The value should
            be arranged from 0 to the number of unique windows
            inds.
    """
    uni_win_inds, _ = torch.sort(torch.unique(win_inds))
    num_valid_inds = len(uni_win_inds)
    max_origin_inds = uni_win_inds.max().item()
    canvas = -win_inds.new_ones((max_origin_inds + 1, ))
    canvas[uni_win_inds] = torch.arange(
        num_valid_inds, dtype=win_inds.dtype, device=win_inds.device)
    conti_inds = canvas[win_inds]
    return conti_inds


def get_seq_to_win_mapping(batch_win_inds, batching_id_per_voxel,
                           region_batching_cfg):
    """Get all transformation info between sequence and batching windows.

    Args:
        batch_win_inds (Tensor): Windows index of each voxel in a batch.
            Voxels share a window if they have same index.
        batching_id_per_voxel (LongTensor): The tensor is the
            batching index of voxel divided by the token number of
            belonging windows.
        region_batching_cfg (list[dict]): Region batching
            configuration. The length of list is the number of
            divisions. The dict has these two keys.

            - max_tokens (int): The number of tokens would be
              padded or clip to.
            - batching_interval (int): The number interval of
              tokens.

    Returns:
        list[tuple]: The list indicate the different batching.
            Each tuple contains three important information
            of corresponding batching.

            - A identical index of voxel in corresponding
              batching windows
            - The index of voxel in original sequence.
            - The number of token of each window in this batching.
    """

    seq_win_mapping_list = []
    for batching_id, single_division_cfg in enumerate(region_batching_cfg):
        division_mask = batching_id_per_voxel == batching_id
        if not division_mask.any():
            continue
        conti_win_inds = make_continuous_inds(batch_win_inds[division_mask])
        max_tokens = region_batching_cfg[batching_id]['max_tokens']
        inner_win_inds = get_inner_win_inds(conti_win_inds)
        seq_window_inds = conti_win_inds * max_tokens + inner_win_inds
        seq_win_mapping_list.append(
            (seq_window_inds, torch.nonzero(division_mask).reshape(-1),
             max_tokens))

    return seq_win_mapping_list


def win_to_seq(region_batching_feat_list, seq_win_mapping_list):
    """Transform the feat from  several batching windows (B, NUM_TOKENS, D) to
    a sequence (N, D).

    Args:
        region_batching_feat_list (list[tensor]): List of
            region bacthing feats, each tensor
            has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
        seq_win_mapping_list (list[tuple]): Each tuple contains
            three important information of corresponding
            batching.

            - A identical index of voxel in corresponding
              batching windows
            - The index of voxel in original sequence.
            - The number of token of each window in this batching.

    Returns:
        Tensor: The feat in sequence format. has shape (N, FEAR_DIM)
    """

    num_voxels = sum([len(item[0]) for item in seq_win_mapping_list])

    feat_dim = region_batching_feat_list[0].size(-1)
    seq_feat = region_batching_feat_list[0].new_zeros((num_voxels, feat_dim))
    for region_batching_feat, seq_win_mapping in zip(region_batching_feat_list,
                                                     seq_win_mapping_list):
        index_in_all_windows, voxel_inds, num_tokens_each_win = seq_win_mapping
        region_batching_feat = region_batching_feat.reshape(-1, feat_dim)
        single_seq_feat = region_batching_feat[index_in_all_windows]
        seq_feat[voxel_inds] = single_seq_feat
    return seq_feat


def seq_to_win(feat, seq_win_mapping_list):
    """Transform the feat from a sequence (N, D) to several batching windows
    (B, NUM_TOKENS, D).

    Args:
        feat (tensor): The sequence features to be transformed.
        seq_win_mapping_list (list[tuple]): Each tuple contains
            three important information of corresponding
            batching.

            - A identical index of voxel in corresponding
              batching windows
            - The index of voxel in original sequence.
            - The number of token of each window in this batching.

    Returns:
        list(tensor): List of region batching feats, each tensor
        has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
    """

    feat_dim = feat.shape[-1]
    region_batching_feat_list = []

    for (index_in_all_windows, voxel_inds,
         num_tokens_each_win) in seq_win_mapping_list:
        feat_in_batching = feat[voxel_inds]
        num_windows = (index_in_all_windows //
                       num_tokens_each_win).max().item() + 1
        region_batching_feat = feat.new_zeros(
            (num_windows * num_tokens_each_win, feat_dim))
        region_batching_feat[index_in_all_windows] = feat_in_batching
        region_batching_feat = region_batching_feat.reshape(
            (num_windows, num_tokens_each_win, feat_dim))
        region_batching_feat_list.append(region_batching_feat)

    return region_batching_feat_list


def get_inner_win_inds(win_inds):
    """Calculate the index of each voxel inner windows.

    The code is modified from original implementation
    https://github.com/TuSimple/SST

    Args:
        win_inds (Tensor): The value indicates which windows a voxel
            belongs to. Voxels share a window have same inds.
            has shape (N, ).

    Return:
        ori_inner_inds (Tensor): The value indicates voxel's id
        in a window. if M voxels share a window, their
        inner_inds would be torch.arange(m, dtype=torch.long)
    """

    # the logic may be complicated
    # We add an example here, win_inds = [0, 1, 2, 0, 0, 1]

    # sort_inds:  [0, 0, 0, 1, 2, 2]
    sort_inds, order = win_inds.sort()

    # roll_inds_left = [0, 0, 1, 2, 2, 0]
    roll_inds_left = torch.roll(sort_inds, -1)

    # diff: [0, 0, -1, -1, 0, 2]
    diff = sort_inds - roll_inds_left

    # end_pos_mask: the indicator of last element in a window
    # [0, 0, 1, 1, 0, 1]
    end_pos_mask = diff != 0

    windows_count = torch.bincount(win_inds)
    unique_sort_inds, _ = torch.sort(torch.unique(win_inds))
    # num_tokens_each_win: [3, 1, 2]
    num_tokens_each_win = windows_count[unique_sort_inds]

    template = torch.ones_like(win_inds)
    # template: [1, 1, -2, 0, 1, -1]
    template[end_pos_mask] = (num_tokens_each_win - 1) * -1

    # inner_inds: [1, 2, 0, 0, 1, 0]
    inner_inds = torch.cumsum(template, 0)
    # inner_inds: [1, 2, 3, 1, 1, 2]
    inner_inds[end_pos_mask] = num_tokens_each_win
    # inner_inds: [0, 1, 2, 0, 0, 1]
    inner_inds -= 1

    # recover the order of original win_inds
    ori_inner_inds = -torch.ones_like(win_inds)
    ori_inner_inds[order] = inner_inds

    return ori_inner_inds
