# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv.runner import auto_fp16
from torch import nn

from mmdet3d.ops.sst_modules import (get_inner_win_inds,
                                     get_seq_to_win_mapping, seq_to_win)
from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class SSTInputLayer(nn.Module):
    """Input layer of SST.

    The code is modified from original implementation
    https://github.com/TuSimple/SST

    This layer would do the regional grouping and region
    batching for sparse regional attention(SRA) in advance and
    pass all related information to sst backbone, all related
    information would be used in SRA of
    SST backbone to do the transformation between the sequence
    format (N, C) and region batching format
    (NUM_WINS_i, NUM_TOKENS_i, C),
    i =0, 1, 2 ,... K, NUM_WINS_i is the number of windows with
    similar number of tokens. NUM_TOKENS_i is the max number of tokens
    in corresponding batching.

    The forward can be divided to 2 steps:
    1. Region Grouping : Assign window indices to each voxel.
    2. Region Batching: Batching the regions with similar
        number of tokens for parallel computation.

    Args:
        region_batching_cfg (list[list[dict]]): Region batching
            configuration. The length if outer list is 2.
            First list will be used in training phase,
            and second list will be used in testing phase.
            The dict contains two keys

            - max_tokens (int): The number of tokens would be
              padded or clip to.
            - batching_interval (int): The number interval of
              tokens.

        window_shape (tuple[int]): (num_x, num_y, num_z). Each window is
            divided to num_x * num_y x num_z pillars
            (including empty pillars). Usually num_z is 1 in SST.
        sparse_shape (tuple[int]): (num_x, num_y). The shape of
            voxel.
        shuffle_voxels (bool): Whether to shuffle the voxels. Defaults
            to True.
        num_attentions (int): The number of attentions is a single
            layer of sst backbone. Defaults to 2.
        embed_temperature (int, optional): The temperature used
            for scaling the position embedding. Defaults to 10000.
        normalize_embed (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
    """

    def __init__(self,
                 region_batching_cfg,
                 window_shape,
                 sparse_shape,
                 shuffle_voxels=True,
                 num_attentions=2,
                 embed_temperature=10000,
                 normalize_embed=False,
                 **kwargs):
        super().__init__()
        self.fp16_enabled = False
        self.region_batching_cfg = region_batching_cfg
        self.window_shape = window_shape
        self.sparse_shape = sparse_shape
        self.shuffle_voxels = shuffle_voxels
        self.num_attentions = num_attentions

        self.embed_temperature = embed_temperature
        self.normalize_embed = normalize_embed

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
    def forward(self, voxel_feats, voxel_coors, *args, **kwargs):
        """Forward of SSTInputLayer.

        Args:
            voxel_feats (tensor): Voxel features in shape (N, C).
            voxel_coors (tensor): Coordinates in shape (N, 4),
                the columns in the order of
                (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            Dict: A dict  contains four element

                - voxel_feats (tensor): Voxel features in shape (N, C).
                - voxel_coors (tensor): Coordinates in shape (N, 4),
                  the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
                - batching_position_embed_list (list[list[tensor]]):
                  The outer list
                  indicate the attentions. The inner list indicate
                  the different batching. Each tensor is the position
                  embedding of corresponding batching. Has shape
                  each tensor has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
                - batching_padding_mask_list (list[list[tensor]]):
                  The outer list indicate the attentions.The inner
                  list indicate the different batching. The tensor is
                  the padding mask of corresponding batching, each
                  tensor has shape (NUM_WINS, NUM_TOKEN).
                - seq_win_mapping_list: (list[list[tuple]]): The outer
                  list indicate the attentions.The inner list indicate
                  the different batching. Each tuple contains 3 important
                  information of corresponding batching.

                    - A identical index of voxel in corresponding
                      batching windows
                    - The index of voxel in original sequence.
                    - The number of token of each window in this batching.
        """
        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_feats))
            voxel_feats = voxel_feats[shuffle_inds]
            voxel_coors = voxel_coors[shuffle_inds]

        batch_win_ind_list, coors_in_win_list = self.region_grouping(
            voxel_coors)

        batch_win_ind_list, batching_id_per_voxel_list, voxel_keep_inds = \
            self.region_batching(batch_win_ind_list)

        voxel_feats = voxel_feats[voxel_keep_inds]
        voxel_coors = voxel_coors[voxel_keep_inds]

        seq_win_mapping_list = []
        batching_padding_mask_list = []
        batching_position_embed_list = []

        region_batching_cfg = self._get_region_batching_cfg()

        feat_dim, feat_dtype = voxel_feats.size(1), voxel_feats.dtype
        for attn_idx in range(self.num_attentions):
            batching_id_per_voxel = batching_id_per_voxel_list[attn_idx]
            batch_win_ind = batch_win_ind_list[attn_idx]
            seq_win_mapping = get_seq_to_win_mapping(batch_win_ind,
                                                     batching_id_per_voxel,
                                                     region_batching_cfg)
            batching_position_embed_list.append(
                self.get_position_embedding(seq_win_mapping,
                                            coors_in_win_list[attn_idx],
                                            feat_dim, feat_dtype))
            batching_padding_mask_list.append(
                self.get_key_padding_mask(batching_id_per_voxel,
                                          seq_win_mapping))

            seq_win_mapping_list.append(seq_win_mapping)
        out_dict = dict()
        out_dict['voxel_feats'] = voxel_feats
        out_dict['voxel_coors'] = voxel_coors
        out_dict['batching_position_embed_list'] = batching_position_embed_list
        out_dict['batching_padding_mask_list'] = batching_padding_mask_list
        out_dict['seq_win_mapping_list'] = batching_padding_mask_list

        return out_dict

    def get_key_padding_mask(self, batching_id_per_voxel, seq_win_mapping):
        """Calculated padding mask for Sparse Region Attention.

        Args:
            batching_id_per_voxel (Tensor): The tensor is the
                batching index of voxel divided by the token number of
                belonging windows.
            seq_win_mapping (list[tuple]): The list indicate the
                different batching. Each tuple Contains three
                important information of corresponding batching.

                - A identical index of voxel in corresponding
                  batching windows
                - The index of voxel in original sequence.
                - The number of token of each window in this batching.

        Returns:
            list(tensor): Key padding mask of each batching.
        """
        num_all_voxel = len(batching_id_per_voxel)
        key_padding = batching_id_per_voxel.new_ones((num_all_voxel, 1),
                                                     dtype=torch.bool)
        all_batching_padding_mask = seq_to_win(key_padding, seq_win_mapping)

        # logical not. True means masked
        all_batching_padding_mask = \
            [item.logical_not().squeeze(2) for item
             in all_batching_padding_mask]
        return all_batching_padding_mask

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
                - voxel_keep_inds (tensor): The remain voxel index after
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

    def get_position_embedding(self, seq_win_mapping, coors_in_win, feat_dim,
                               dtype):
        """Calculate the position embedding for Sparse Region Attention.

        Args:
            seq_win_mapping (list[tuple]): The list indicate the
                different batching. Each tuple contains 3 important
                information of corresponding batching.

                  - A identical index of voxel in corresponding
                    batching windows
                  - The index of voxel in original sequence.
                  - The number of token of each window in this batching.
            coors_in_win (Tensor): The tensor is the coordinate of voxel
                  in a windows.
            feat_dim (int): The dimension of feature.
            dtype (torch.dtype): The dtype of feature.

        Returns:
            list[tensor]: List of region batching
            position embedding, each tensor
            has shape (NUM_WINS, NUM_TOKEN, FEAT_DIM).
        """

        window_shape = self.window_shape
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

        x = coors_in_win[:, 2] - win_x / 2
        y = coors_in_win[:, 1] - win_y / 2
        z = coors_in_win[:, 0] - win_z / 2

        if self.normalize_embed:
            x = x / win_x * 2 * math.pi
            y = y / win_y * 2 * math.pi
            z = z / win_z * 2 * math.pi

        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.embed_temperature**(2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()],
                              dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()],
                              dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack(
                [embed_z[:, ::2].sin(), embed_z[:, 1::2].cos()],
                dim=-1).flatten(1)

        if ndim == 3:
            position_embed_2d = torch.cat([embed_x, embed_y, embed_z],
                                          dim=-1).to(dtype)
        else:
            position_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)

        gap = feat_dim - position_embed_2d.size(1)
        assert gap >= 0
        if gap > 0:
            assert ndim == 3
            padding = torch.zeros((position_embed_2d.size(0), gap),
                                  dtype=dtype,
                                  device=coors_in_win.device)
            position_embed_2d = torch.cat([position_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        all_batching_position_embeding = seq_to_win(position_embed_2d,
                                                    seq_win_mapping)

        return all_batching_position_embeding

    def _single_region_batching(self, batch_win_inds):
        """Do the region batching for a single attention.

        Args:
            batch_win_inds (Tensor): Windows index of each voxel in a batch.
            Voxels share a window if they have same index.

        Returns:
            Tuple[Tensor]:

                - keep_mask (Tensor): The mask of remain voxels
                  after drop
                - batching_id_per_voxel (Tensor): The tensor is the
                  batching index of voxel divided by the token number of
                  belonging windows.
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
           voxel_coors (tensor): Coordinates in shape (N, 4), \
            the columns in the order of (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            tuple[Tensor]:

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
           coors (tensor): Coordinates in shape (N, 4), \
               the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
           do_shift (bool): Whether do the grouping for the shift window
               attention.

        Returns:
            tuple:

                - batch_win_inds (tensor): The window index of each
                  voxel in a batch.
                - coors_in_win (tensor): The coordinate of voxel
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
            shift_x, shift_y, shift_z = \
                win_shape_x, win_shape_y, win_shape_z

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
