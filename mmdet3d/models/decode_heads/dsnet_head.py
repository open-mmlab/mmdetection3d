# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_conv_layer
from torch import nn

from mmdet3d.ops import make_sparse_convmodule
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseSequential
else:
    from mmcv.ops import SparseSequential


@HEADS.register_module()
class DSNetSemHead(Base3DDecodeHead):

    def __init__(self,
                 nclasses,
                 conv_cfg=dict(type='SubMConv3d', indice_key='logit'),
                 init_size=32,
                 **kwargs):
        super(DSNetSemHead, self).__init__(**kwargs)

        self.logits = build_conv_layer(
            conv_cfg, init_size, nclasses, 3, stride=1, padding=1, bias=False)

    def forward(self, fea):
        logits = self.logits(fea)
        return logits.dense()


@HEADS.register_module()
class DSNetInstHead(Base3DDecodeHead):

    def __init__(self,
                 init_size=32,
                 embedding_dim=3,
                 indice_keys=('offset_head_conv1', ),
                 conv_type='SubMConv3d',
                 paddings=(1, ),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 act_cfg=dict(type='LeakyReLU'),
                 **kwargs):
        super(DSNetInstHead, self).__init__(**kwargs)

        self.pt_fea_dim = 4 * init_size
        self.encoder_channels = ((self.pt_fea_dim, 2 * init_size, init_size), )
        self.encoder_paddings = paddings
        self.inst_conv = self.make_encoder_layers(make_sparse_convmodule,
                                                  indice_keys, norm_cfg,
                                                  act_cfg, self.pt_fea_dim)

        self.offset = nn.Sequential(
            nn.Linear(init_size + 3, init_size, bias=True),
            nn.BatchNorm1d(init_size), nn.ReLU())
        self.offset_linear = nn.Linear(init_size, embedding_dim, bias=True)

    def forward(self, fea, input, prefix=''):
        fea = self.inst_conv(fea)

        grid_ind = input[prefix + 'grid']
        xyz = input[prefix + 'pt_cart_xyz']
        fea = fea.dense()
        fea = fea.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(fea[batch_i, grid_ind[batch_i][:, 0],
                                       grid_ind[batch_i][:, 1],
                                       grid_ind[batch_i][:, 2]])
        pt_pred_offsets_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_offsets_list.append(
                self.offset_linear(
                    self.offset(
                        torch.cat([
                            pt_ins_fea,
                            torch.from_numpy(xyz[batch_i]).cuda()
                        ],
                                  dim=1))))
        return pt_pred_offsets_list, pt_ins_fea_list

    def make_encoder_layers(self, make_block, indice_keys, norm_cfg, act_cfg,
                            in_channels):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.

        Returns:
            int: The number of encoder output channels.
        """
        encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]

                blocks_list.append(
                    make_block(
                        in_channels,
                        out_channels,
                        3,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        padding=padding,
                        indice_key=indice_keys[j],
                        conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            encoder_layers.add_module(stage_name, stage_layers)
        return encoder_layers
