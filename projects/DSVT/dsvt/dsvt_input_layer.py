# modified from https://github.com/Haiyang-W/DSVT
from math import ceil

import torch
from torch import nn

from .utils import (PositionEmbeddingLearned, get_continous_inds,
                    get_inner_win_inds_cuda, get_pooling_index,
                    get_window_coors)


class DSVTInputLayer(nn.Module):
    '''
    This class converts the output of vfe to dsvt input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets
        within each window.
    3. Pre-compute the downsample information between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space
            (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes
            (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample
            strides between two consecutive stages.
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and
            stage_{i+1}. Length: stage_num - 1.
        dim_model (list[int]): Number of input channels for each stage. Length:
            stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage.
            Eelement i contains
            [set_size, block_num], where set_size is the number of voxel in a
            set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in
            different blocks.
            e.g. for block_{0} and block_{1} in stage_0, window shapes are
            [win_x, win_y, win_z] and
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        normalize_pos (bool): Whether to normalize coordinates in position
            embedding.
    '''

    def __init__(self, sparse_shape, window_shape, downsample_stride,
                 dim_model, set_info, hybrid_factor, shift_list,
                 normalize_pos):
        super().__init__()

        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.downsample_stride = downsample_stride
        self.dim_model = dim_model
        self.set_info = set_info
        self.stage_num = len(self.dim_model)

        self.hybrid_factor = hybrid_factor
        self.window_shape = [[
            self.window_shape[s_id],
            [
                self.window_shape[s_id][coord_id] *
                self.hybrid_factor[coord_id] for coord_id in range(3)
            ]
        ] for s_id in range(self.stage_num)]
        self.shift_list = shift_list
        self.normalize_pos = normalize_pos

        self.num_shifts = [
            2,
        ] * len(self.window_shape)

        self.sparse_shape_list = [self.sparse_shape]
        # compute sparse shapes for each stage
        for ds_stride in self.downsample_stride:
            last_sparse_shape = self.sparse_shape_list[-1]
            self.sparse_shape_list.append(
                (ceil(last_sparse_shape[0] / ds_stride[0]),
                 ceil(last_sparse_shape[1] / ds_stride[1]),
                 ceil(last_sparse_shape[2] / ds_stride[2])))

        # position embedding layers
        self.posembed_layers = nn.ModuleList()
        for i in range(len(self.set_info)):
            input_dim = 3 if self.sparse_shape_list[i][-1] > 1 else 2
            stage_posembed_layers = nn.ModuleList()
            for j in range(self.set_info[i][1]):
                block_posembed_layers = nn.ModuleList()
                for s in range(self.num_shifts[i]):
                    block_posembed_layers.append(
                        PositionEmbeddingLearned(input_dim, self.dim_model[i]))
                stage_posembed_layers.append(block_posembed_layers)
            self.posembed_layers.append(stage_posembed_layers)

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict):
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE
                    with shape (N, dim_model[0]),
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding
                    voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x).
                - ...

        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is
                    the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition
                    index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition.
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask
                    used in set attention with shape
                    (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]):
                    Position embedding vectors with shape (N_i, dim_model[i]).
                    N_i is the number of remain voxels in stage_i;
                - pooling_mapping_index_stage{i} (Tensor[int]): Pooling region
                    index used in pooling operation between stage_{i-1}
                    and stage_{i} with shape (N_{i-1}).
                - pooling_index_in_pool_stage{i} (Tensor[int]): Index inner
                    region with shape (N_{i-1}). Combined with
                    pooling_mapping_index_stage{i}, we can map each voxel in
                    satge_{i-1} to pooling_preholder_feats_stage{i}, which
                    are input of downsample operation.
                - pooling_preholder_feats_stage{i} (Tensor[int]): Preholder
                    features initial with value 0.
                    Shape of (N_{i}, downsample_stride[i-1].prob(),
                    d_moel[i-1]), where prob() returns the product of
                    all elements.
                - ...
        '''
        voxel_feats = batch_dict['voxel_features']
        voxel_coors = batch_dict['voxel_coords'].long()

        voxel_info = {}
        voxel_info['voxel_feats_stage0'] = voxel_feats.clone()
        voxel_info['voxel_coors_stage0'] = voxel_coors.clone()

        for stage_id in range(self.stage_num):
            # window partition of corresponding stage-map
            voxel_info = self.window_partition(voxel_info, stage_id)
            # generate set id of corresponding stage-map
            voxel_info = self.get_set(voxel_info, stage_id)
            for block_id in range(self.set_info[stage_id][1]):
                for shift_id in range(self.num_shifts[stage_id]):
                    layer_name = f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'  # noqa: E501
                    pos_name = f'coors_in_win_stage{stage_id}_shift{shift_id}'
                    voxel_info[layer_name] = self.get_pos_embed(
                        voxel_info[pos_name], stage_id, block_id, shift_id)

            # compute pooling information
            if stage_id < self.stage_num - 1:
                voxel_info = self.subm_pooling(voxel_info, stage_id)

        return voxel_info

    @torch.no_grad()
    def subm_pooling(self, voxel_info, stage_id):
        # x,y,z stride
        cur_stage_downsample = self.downsample_stride[stage_id]
        # batch_win_coords is from 1 of x, y
        batch_win_inds, _, index_in_win, batch_win_coors = get_pooling_index(
            voxel_info[f'voxel_coors_stage{stage_id}'],
            self.sparse_shape_list[stage_id], cur_stage_downsample)
        # compute pooling mapping index
        unique_batch_win_inds, contiguous_batch_win_inds = torch.unique(
            batch_win_inds, return_inverse=True)
        voxel_info[
            f'pooling_mapping_index_stage{stage_id+1}'] = \
            contiguous_batch_win_inds

        # generate empty placeholder features
        placeholder_prepool_feats = voxel_info['voxel_feats_stage0'].new_zeros(
            (len(unique_batch_win_inds),
             torch.prod(torch.IntTensor(cur_stage_downsample)).item(),
             self.dim_model[stage_id]))
        voxel_info[f'pooling_index_in_pool_stage{stage_id+1}'] = index_in_win
        voxel_info[
            f'pooling_preholder_feats_stage{stage_id+1}'] = \
            placeholder_prepool_feats

        # compute pooling coordinates
        unique, inverse = unique_batch_win_inds.clone(
        ), contiguous_batch_win_inds.clone()
        perm = torch.arange(
            inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        pool_coors = batch_win_coors[perm]

        voxel_info[f'voxel_coors_stage{stage_id+1}'] = pool_coors

        return voxel_info

    def get_set(self, voxel_info, stage_id):
        '''
        This is one of the core operation of DSVT.
        Given voxels' window ids and relative-coords inner window, we partition
        them into window-bounded and size-equivalent local sets. To make it
        clear and easy to follow, we do not use loop to process two shifts.
        Args:
            voxel_info (dict):
                The dict contains the following keys
                - batch_win_inds_s{i} (Tensor[float]): Windows indices of each
                    voxel with shape (N), computed by 'window_partition'.
                - coors_in_win_shift{i} (Tensor[int]): Relative-coords inner
                    window of each voxel with shape (N, 3), computed by
                    'window_partition'. Each row is (z, y, x).
                - ...

        Returns:
            See from 'forward' function.
        '''
        batch_win_inds_shift0 = voxel_info[
            f'batch_win_inds_stage{stage_id}_shift0']
        coors_in_win_shift0 = voxel_info[
            f'coors_in_win_stage{stage_id}_shift0']
        set_voxel_inds_shift0 = self.get_set_single_shift(
            batch_win_inds_shift0,
            stage_id,
            shift_id=0,
            coors_in_win=coors_in_win_shift0)
        voxel_info[
            f'set_voxel_inds_stage{stage_id}_shift0'] = set_voxel_inds_shift0
        # compute key masks, voxel duplication must happen continuously
        prefix_set_voxel_inds_s0 = torch.roll(
            set_voxel_inds_shift0.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s0[:, :, 0] = -1
        set_voxel_mask_s0 = (set_voxel_inds_shift0 == prefix_set_voxel_inds_s0)
        voxel_info[
            f'set_voxel_mask_stage{stage_id}_shift0'] = set_voxel_mask_s0

        batch_win_inds_shift1 = voxel_info[
            f'batch_win_inds_stage{stage_id}_shift1']
        coors_in_win_shift1 = voxel_info[
            f'coors_in_win_stage{stage_id}_shift1']
        set_voxel_inds_shift1 = self.get_set_single_shift(
            batch_win_inds_shift1,
            stage_id,
            shift_id=1,
            coors_in_win=coors_in_win_shift1)
        voxel_info[
            f'set_voxel_inds_stage{stage_id}_shift1'] = set_voxel_inds_shift1
        # compute key masks, voxel duplication must happen continuously
        prefix_set_voxel_inds_s1 = torch.roll(
            set_voxel_inds_shift1.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s1[:, :, 0] = -1
        set_voxel_mask_s1 = (set_voxel_inds_shift1 == prefix_set_voxel_inds_s1)
        voxel_info[
            f'set_voxel_mask_stage{stage_id}_shift1'] = set_voxel_mask_s1

        return voxel_info

    def get_set_single_shift(self,
                             batch_win_inds,
                             stage_id,
                             shift_id=None,
                             coors_in_win=None):
        device = batch_win_inds.device
        # the number of voxels assigned to a set
        voxel_num_set = self.set_info[stage_id][0]
        # max number of voxels in a window
        max_voxel = self.window_shape[stage_id][shift_id][
            0] * self.window_shape[stage_id][shift_id][1] * self.window_shape[
                stage_id][shift_id][2]
        # get unique set indices
        contiguous_win_inds = torch.unique(
            batch_win_inds, return_inverse=True)[1]
        voxelnum_per_win = torch.bincount(contiguous_win_inds)
        win_num = voxelnum_per_win.shape[0]
        setnum_per_win_float = voxelnum_per_win / voxel_num_set
        setnum_per_win = torch.ceil(setnum_per_win_float).long()
        set_win_inds, set_inds_in_win = get_continous_inds(setnum_per_win)

        # compution of Eq.3 in 'DSVT: Dynamic Sparse Voxel Transformer with
        # Rotated Sets' - https://arxiv.org/abs/2301.06051,
        # for each window, we can get voxel indices belong to different sets.
        offset_idx = set_inds_in_win[:, None].repeat(
            1, voxel_num_set) * voxel_num_set
        base_idx = torch.arange(0, voxel_num_set, 1, device=device)
        base_select_idx = offset_idx + base_idx
        base_select_idx = base_select_idx * voxelnum_per_win[
            set_win_inds][:, None]
        base_select_idx = base_select_idx.double() / (
            setnum_per_win[set_win_inds] * voxel_num_set)[:, None].double()
        base_select_idx = torch.floor(base_select_idx)
        # obtain unique indices in whole space
        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel

        # this function will return unordered inner window indices of
        # each voxel
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds)
        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds
        _, order1 = torch.sort(global_voxel_inds)

        # get y-axis partition results
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * \
            self.window_shape[stage_id][shift_id][2] + coors_in_win[:, 2] * \
            self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sorty)
        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1]
        )  # get y-axis ordered inner window indices of each voxel
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty + max_voxel * \
            contiguous_win_inds
        voxel_inds_padding_sorty = -1 * torch.ones(
            (win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(
            0,
            voxel_inds_in_batch_sorty.shape[0],
            dtype=torch.long,
            device=device)
        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()]

        # get x-axis partition results
        global_voxel_inds_sortx = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * \
            self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sortx)
        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1]
        )  # get x-axis ordered inner window indices of each voxel
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx + max_voxel * \
            contiguous_win_inds
        voxel_inds_padding_sortx = -1 * torch.ones(
            (win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(
            0,
            voxel_inds_in_batch_sortx.shape[0],
            dtype=torch.long,
            device=device)
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]

        all_set_voxel_inds = torch.stack(
            (set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0)
        return all_set_voxel_inds

    @torch.no_grad()
    def window_partition(self, voxel_info, stage_id):
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(
                voxel_info[f'voxel_coors_stage{stage_id}'],
                self.sparse_shape_list[stage_id],
                self.window_shape[stage_id][i], i == 1,
                self.shift_list[stage_id][i])

            voxel_info[
                f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win

        return voxel_info

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
            coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id]

        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z / 2,\
            coors_in_win[:, 1] - win_y / 2,\
            coors_in_win[:, 2] - win_x / 2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
