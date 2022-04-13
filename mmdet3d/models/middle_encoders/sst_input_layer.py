# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn
import numpy as np
from ..builder import MIDDLE_ENCODERS
from mmdet3d.ops.sst_modules import (get_inner_win_inds, get_flat2win_inds)


@MIDDLE_ENCODERS.register_module()
class SSTInputLayer(nn.Module):
    """Input layer of SST.

    This layer would do the regional grouping and region
    grouping for sparse regional attention(SRA) in advance and
    save all related information to a dict and pass it
    to SST backbone. This dict can be used in SRA of
    SST backbone to do the transformation between the sequence
    format (N, C) and region batching format (B_i, N_i, C),
    i =0, 1, 2 ,... K, B_i is the number of windows with
    similar number of tokens. The code is modified from
    the offitial implementation

    The forward can be divided to 2 steps:
    1. Reginal Grouping : Assign window indices to each voxel.
    2. Region Batching: Batching the regions with similar
        number of tokens for parallel computation.

    Args:
        region_batching_cfg (list[list[dict]]): Region batching
            configuration. The length if outer list is 2.
            First list will be used in training phase,
            and second list will be used in testing phase.
            The dict contains two keys

            - max_tokens (int): The number of tokens would be
              padded or cliped to.
            - batching_interval (int): The number interval of
              tokens.

        window_shape (tuple[int]): (num_x, num_y). Each window is
            divided to num_x * num_y pillars (including empty pillars).
        sparse_shape (tuple[int]): (num_x, num_y). The shape of
            voxel.
        shuffle_voxels (bool): Whether to shuffle the voxels. Defaults
            to True.
        num_attentions (int): The number of attentions is a single
            layer of sst backbone. Defaults to 2.
    """

    def __init__(
        self,
        region_batching_cfg,
        window_shape,
        sparse_shape,
        shifts_list=None,
        shuffle_voxels=True,
        num_attentions=2,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.region_batching_cfg = region_batching_cfg
        self.window_shape = window_shape
        self.sparse_shape = sparse_shape
        self.shuffle_voxels = shuffle_voxels
        self.shifts_list = shifts_list
        self.num_attentions = num_attentions

    def _get_region_batching_cfg(self):
        """ Get region batching configuration.
        Returns:
            List(dict): The dict contains following keys:

            - max_tokens (int): The number of tokens would be
              padded or cliped to.
            - batching_interval (int): The number interval of
              tokens.
        """
        if hasattr(self, '_region_batching_cfg'):
            return self._region_batching_cfg
        meta = self.region_batching_cfg
        if isinstance(meta, tuple):
            if self.training:
                self._region_batching_cfg = meta[0]
            else:
                self._region_batching_cfg = meta[1]
        else:
            self._region_batching_cfg = meta
        return self._region_batching_cfg

    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, **kwargs):
        """Forward of SSTInputLayer.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            voxel_coors (torch.float32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            tuple:

                - voxel_feats (torch.float32): Voxel features in shape (N, C).
                - voxel_coors (torch.int32): Coordinates in shape (N, 4), \
                    the columns in the order of
                    (batch_idx, z_idx, y_idx, x_idx).
                Todo design this filed afte refactor the sst backbone
                -
                -
        """
        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_feats))
            voxel_feats = voxel_feats[shuffle_inds]
            voxel_coors = voxel_coors[shuffle_inds]
        else:
            shuffle_inds = None

        batch_win_ind_list, coors_in_win_list = self.region_grouping(
            voxel_coors)

        batch_win_ind_list, batching_id_per_voxel_list, voxel_keep_inds = \
            self.region_batching(batch_win_ind_list, coors_in_win_list)

        voxel_feats = voxel_feats[voxel_keep_inds]  # after dropping
        voxel_coors = voxel_coors[voxel_keep_inds]
        seq_win_mapping_list = []
        position_embedding_list = []
        key_mask_list = []
        region_batching_cfg = self._get_region_batching_cfg()
        for attn_idx in range(self.num_attentions):
            # TODO refactor the get_flat2win_inds
            seq_win_mapping = get_flat2win_inds(
                batch_win_ind_list[attn_idx],
                batching_id_per_voxel_list[attn_idx], region_batching_cfg)
            seq_win_mapping_list.append(seq_win_mapping)
            position_embedding = self.get_pos_embed(
                seq_win_mapping, coors_in_win_list[attn_idx],
                voxel_feats.size(1), voxel_feats.dtype)
            position_embedding_list.append(position_embedding)

            key_mask = self.get_key_padding_mask(seq_win_mapping)
            key_mask_list.append(key_mask)

        return (voxel_feats, voxel_coors, position_embedding_list,
                key_mask_list, seq_win_mapping_list, shuffle_inds)

    def region_batching(self, batch_win_ind_list):
        """Batching the regions with similar number of tokens for parallel
        computation.

        Args:
            batch_win_ind_list (list[tensor]): The outer list indicate
                the attentions. The tensor is the window index of each
                voxel in a batch before region batching.

        Returns:
            tuple:

                - batch_win_ind_list (list[tensor]): The outer list indicate
                  the attentions. The tensor is the window index of each
                  voxel in a batch before region batching.
                - batching_id_per_voxel_list (list[tensor]): The outer list
                  indicate the attentions. The tensor is the batching index
                  divided by the  number of tokens.
                - voxel_keep_inds (torch.int64): The remain voxel index after
                  region batching process of all attentions.
        """
        num_voxels = batch_win_ind_list[0].shape[0]
        voxel_keep_inds = torch.arange(
            num_voxels, device=batch_win_ind_list[0].device, dtype=torch.long)

        batching_id_per_voxel_list = []

        # drop the voxels attn by attn
        for attn_idx in range(self.num_attentions):
            batch_win_inds = batch_win_ind_list[attn_idx]
            keep_mask, batching_id_per_voxel = self._single_region_batching(
                batch_win_inds)
            batching_id_per_voxel_list.append(batching_id_per_voxel)

            # do the drop for all attention
            batching_id_per_voxel_list = [
                item[keep_mask] for item in batching_id_per_voxel_list
            ]
            batch_win_ind_list = [
                item[keep_mask] for item in batch_win_ind_list
            ]
            voxel_keep_inds = voxel_keep_inds[keep_mask]

        return batch_win_ind_list, batching_id_per_voxel_list, voxel_keep_inds

    def _single_region_batching(self, batch_win_inds):
        """Do the region batching for a single attention.

        Args:
            batch_win_inds (Tensor): Windows index of each voxel in a batch.
            Voxels share a window if they have same index.

        Returns:
            Tuple[Tensor]:

                - keep_mask (BoolTensor): The mask of remain voxels
                  after drop
                - batching_id_per_voxel (LongTensor): The level of number
                  of token belonging to.
        """
        region_batching_cfg = self._get_region_batching_cfg()
        batching_id_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        win_num_tokens_per_voxel = bincount[batch_win_inds]  #
        max_index_per_voxel = torch.zeros_like(batch_win_inds)

        for batching_id, batching_info in enumerate(region_batching_cfg):
            max_tokens = batching_info['max_tokens']
            lower, upper = batching_info['batching_interval']
            range_mask = (win_num_tokens_per_voxel >= lower) & (
                win_num_tokens_per_voxel < upper)
            max_index_per_voxel[range_mask] = max_tokens
            batching_id_per_voxel[range_mask] = batching_id

        keep_mask = inner_win_inds < max_index_per_voxel
        return keep_mask, batching_id_per_voxel

    def region_grouping(self, voxel_coors):
        """Divide the voxel to different windows.

        Args:
           voxel_coors (torch.float32): Coordinates in shape (N, 4), \
            the columns in the order of (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            tuple (Tensor):

                - batch_win_ind_list (list[tensor]): The outer list indicate
                    the attentions. The tensor is the window index of each
                    voxel in a batch. Voxels share a window if they have
                    same index.
                - coors_in_win_list (list[tensor]): The outer list indicate
                    the attentions. The tensor is the coordinate of voxel
                    in a windows.
        """
        batch_win_ind_list = []
        coors_in_win_list = []
        for attn_idx in range(self.num_attentions):
            batch_win_ind, coors_in_win = self._single_region_grouping(
                voxel_coors, attn_idx != 0)
            batch_win_ind_list.append(batch_win_ind)
            coors_in_win_list.append(coors_in_win)

        return batch_win_ind_list, coors_in_win_list

    def _single_region_grouping(self, coors, do_shift):
        """Do the region grouping for single attentions.
        Args:
           coors (torch.float32): Coordinates in shape (N, 4), \
               the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
           do_shift (bool): Whether do the grouping for the shift window
               attention.

        Returns:
            batch_win_inds (torch.int64): The window index of each
            voxel in a batch.
            coors_in_win (torch.int64): The coordinate of voxel
            in a windows.
        """
        sparse_shape = self.sparse_shape
        window_shape = self.window_shape
        coors = coors.long()
        if len(window_shape) == 2:
            win_shape_x, win_shape_y = window_shape
            win_shape_z = sparse_shape[-1]
        else:
            win_shape_x, win_shape_y, win_shape_z = window_shape

        sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape

        # Add extra 1 here to meet the needs of shift with  0.5 * windows_size
        max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)
        max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)
        max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)
        max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

        if do_shift:
            shift_x, shift_y, shift_z = \
                win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
        else:
            shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z

        # compatibility between 2D window and 3D window
        if sparse_shape_z == win_shape_z:
            shift_z = 0

        shifted_coors_x = coors[:, 3] + shift_x
        shifted_coors_y = coors[:, 2] + shift_y
        shifted_coors_z = coors[:, 1] + shift_z

        win_coors_x = shifted_coors_x // win_shape_x
        win_coors_y = shifted_coors_y // win_shape_y
        win_coors_z = shifted_coors_z // win_shape_z

        # coors[:, 0] is the batch index
        batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
            win_coors_x * max_num_win_y * max_num_win_z + \
            win_coors_y * max_num_win_z + \
            win_coors_z

        coors_in_win_x = shifted_coors_x % win_shape_x
        coors_in_win_y = shifted_coors_y % win_shape_y
        coors_in_win_z = shifted_coors_z % win_shape_z
        coors_in_win = torch.stack(
            [coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

        return batch_win_inds, coors_in_win

    def get_pos_embed(self, ):
        # todo waiting for refacor
        return None

    def get_key_padding_mask(self, ):
        # todo waiting for refacor
        return None
