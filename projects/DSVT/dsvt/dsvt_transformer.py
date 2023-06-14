# modified from https://github.com/Haiyang-W/DSVT
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS
from .dsvt_input_layer import DSVTInputLayer


@MODELS.register_module()
class DSVTMiddleEncoder(nn.Module):
    '''Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe
            to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length:
            stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage.
            Eelement i contains
            [set_size, block_num], where set_size is the number of voxel in a
            set and block_num is the number of blocks for stage i. Length:
            stage_num.
        dim_model (list[int]): Number of input channels for each stage.
            Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage.
            Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in
            set attention for each stage. Length: stage num.
        dropout (float): Drop rate of set attention.
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages.
            One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.

    '''

    def __init__(
            self,
            input_layer=dict(
                sparse_shape=[468, 468, 1],
                downsample_stride=[],
                dim_model=[192],
                set_info=[[36, 4]],
                window_shape=[[12, 12, 1]],
                hybrid_factor=[2, 2, 1],  # x, y, z
                shifts_list=[[[0, 0, 0], [6, 6, 0]]],
                normalize_pos=False),
            stage_num=1,
            output_shape=[468, 468],
            reduction_type='attention',
            downsample_stride=[],
            set_info=[[36, 4]],
            dim_model=[192],
            dim_feedforward=[384],
            nhead=[8],
            conv_out_channel=192,
            dropout=0.,
            activation='gelu'):
        super().__init__()
        self.input_layer = DSVTInputLayer(**input_layer)
        self.reduction_type = reduction_type

        # Sparse Regional Attention Blocks
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = dim_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_list = []
            norm_list = []
            for i in range(num_blocks_this_stage):
                block_list.append(
                    DSVTBlock(
                        dmodel_this_stage,
                        num_head_this_stage,
                        dfeed_this_stage,
                        dropout,
                        activation,
                        batch_first=True))
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}',
                             nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num - 1:
                downsample_window = downsample_stride[stage_id]
                dmodel_next_stage = dim_model[stage_id + 1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(
                        downsample_window).prod().item()
                    self.__setattr__(
                        f'stage_{stage_id}_reduction',
                        StageReductionBlock(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction',
                                     torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(
                        f'stage_{stage_id}_reduction',
                        StageReductionAttBlock(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError

        self.num_shifts = [2] * stage_num
        self.output_shape = output_shape
        self.stage_num = stage_num
        self.set_info = set_info
        self.num_point_features = conv_out_channel

        self._reset_parameters()

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict):
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE.
                    Shape of (N, dim_model[0]),
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding
                    voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x).
                - ...

        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        '''
        voxel_info = self.input_layer(batch_dict)

        voxel_feat = voxel_info['voxel_feats_stage0']
        set_voxel_inds_list = [[
            voxel_info[f'set_voxel_inds_stage{s}_shift{i}']
            for i in range(self.num_shifts[s])
        ] for s in range(self.stage_num)]
        set_voxel_masks_list = [[
            voxel_info[f'set_voxel_mask_stage{s}_shift{i}']
            for i in range(self.num_shifts[s])
        ] for s in range(self.stage_num)]
        pos_embed_list = [[[
            voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}']
            for i in range(self.num_shifts[s])
        ] for b in range(self.set_info[s][1])] for s in range(self.stage_num)]
        pooling_mapping_index = [
            voxel_info[f'pooling_mapping_index_stage{s+1}']
            for s in range(self.stage_num - 1)
        ]
        pooling_index_in_pool = [
            voxel_info[f'pooling_index_in_pool_stage{s+1}']
            for s in range(self.stage_num - 1)
        ]
        pooling_preholder_feats = [
            voxel_info[f'pooling_preholder_feats_stage{s+1}']
            for s in range(self.stage_num - 1)
        ]

        output = voxel_feat
        block_id = 0
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(
                f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                output = block(
                    output,
                    set_voxel_inds_list[stage_id],
                    set_voxel_masks_list[stage_id],
                    pos_embed_list[stage_id][i],
                    block_id=block_id)
                output = residual_norm_layers[i](output + residual)
                block_id += 1
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats[stage_id].type_as(
                    output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index[stage_id],
                                 pooling_index_in_pool[stage_id]] = output
                prepool_features = prepool_features.view(
                    prepool_features.shape[0], -1)

                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(
                        prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(
                        pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(
                        prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(
                        pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros(
                        (pooled_voxel_num,
                         pool_volume)).to(prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(
                        prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output
        batch_dict['voxel_coords'] = voxel_info[
            f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)


class DSVTBlock(nn.Module):
    """Consist of two encoder layer, shift and shift back."""

    def __init__(self,
                 dim_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 batch_first=True):
        super().__init__()

        encoder_1 = DSVTEncoderLayer(dim_model, nhead, dim_feedforward,
                                     dropout, activation, batch_first)
        encoder_2 = DSVTEncoderLayer(dim_model, nhead, dim_feedforward,
                                     dropout, activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
        self,
        src,
        set_voxel_inds_list,
        set_voxel_masks_list,
        pos_embed_list,
        block_id,
    ):
        num_shifts = 2
        output = src
        for i in range(num_shifts):
            set_id = i
            shift_id = block_id % 2
            pos_embed_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[pos_embed_id]
            layer = self.encoder_list[i]
            output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)

        return output


class DSVTEncoderLayer(nn.Module):

    def __init__(self,
                 dim_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 batch_first=True,
                 mlp_dropout=0):
        super().__init__()
        self.win_attn = SetAttention(dim_model, nhead, dropout,
                                     dim_feedforward, activation, batch_first,
                                     mlp_dropout)
        self.norm = nn.LayerNorm(dim_model)
        self.dim_model = dim_model

    def forward(self, src, set_voxel_inds, set_voxel_masks, pos=None):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds)
        src = src + identity
        src = self.norm(src)

        return src


class SetAttention(nn.Module):

    def __init__(self,
                 dim_model,
                 nhead,
                 dropout,
                 dim_feedforward=2048,
                 activation='relu',
                 batch_first=True,
                 mlp_dropout=0):
        super().__init__()
        self.nhead = nhead
        if batch_first:
            self.self_attn = nn.MultiheadAttention(
                dim_model, nhead, dropout=dropout, batch_first=batch_first)
        else:
            self.self_attn = nn.MultiheadAttention(
                dim_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.dim_model = dim_model
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None):
        '''
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is
                the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels
                within set. Shape of (set_num, set_size).
            voxel_inds (Tensor[int]): Voxel indices for each set.
                Shape of (set_num, set_size).
        Returns:
            src (Tensor[float]): Voxel features.
        '''
        set_features = src[voxel_inds]
        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None
        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features

        if key_padding_mask is not None:
            src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]

        # map voxel features from set space to voxel space:
        # (set_num, set_size, C) --> (N, C)
        flatten_inds = voxel_inds.reshape(-1)
        unique_flatten_inds, inverse = torch.unique(
            flatten_inds, return_inverse=True)
        perm = torch.arange(
            inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(
            0, inverse, perm)
        src2 = src2.reshape(-1, self.dim_model)[perm]

        # FFN layer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class StageReductionBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, output_channel, bias=False)
        self.norm = nn.LayerNorm(output_channel)

    def forward(self, x):
        src = x
        src = self.norm(self.linear1(x))
        return src


class StageReductionAttBlock(nn.Module):

    def __init__(self, input_channel, pool_volume):
        super().__init__()
        self.pool_volume = pool_volume
        self.query_func = torch.nn.MaxPool1d(pool_volume)
        self.norm = nn.LayerNorm(input_channel)
        self.self_attn = nn.MultiheadAttention(
            input_channel, 8, batch_first=True)
        self.pos_embedding = nn.Parameter(
            torch.randn(pool_volume, input_channel))
        nn.init.normal_(self.pos_embedding, std=.01)

    def forward(self, x, key_padding_mask):
        # x: [voxel_num, c_dim, pool_volume]
        src = self.query_func(x).permute(0, 2, 1)  # voxel_num, 1, c_dim
        key = value = x.permute(0, 2, 1)
        key = key + self.pos_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1)
        query = src.clone()
        output = self.self_attn(query, key, value, key_padding_mask)[0]
        src = self.norm(output + src).squeeze(1)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return torch.nn.functional.relu
    if activation == 'gelu':
        return torch.nn.functional.gelu
    if activation == 'glu':
        return torch.nn.functional.glu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')
