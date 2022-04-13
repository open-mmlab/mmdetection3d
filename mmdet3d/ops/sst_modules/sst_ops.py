# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_inner_win_inds(win_inds):
    """Calculate the index of each voxel inner windows.

    Note that this function might output different results
    from get_inner_win_inds_slow due to the unstable pytorch sort.

    Args:
        win_inds (tensor): The value indicates which windows a
            voxel belongs to. Voxels share a window have same
            inds, has shape (N,).
    Return:
        ori_inner_inds (tensor): Has shape (N,). Indicates
        voxel's id in a window. if M voxels share a window, their
        inner_inds would be torch.arange(m, dtype=torch.long)
    """

    # the logic may be comlicated
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
