import torch.nn as nn

import mmdet3d.ops.spconv as spconv
from ..registry import MIDDLE_ENCODERS
from ..utils import build_norm_layer


@MIDDLE_ENCODERS.register_module
class SparseEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 output_shape,
                 pre_act,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.sparse_shape = output_shape
        self.output_shape = output_shape
        self.in_channels = in_channels
        self.pre_act = pre_act
        # Spconv init all weight on its own
        # TODO: make the network could be modified

        if pre_act:
            self.conv_input = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    16,
                    3,
                    padding=1,
                    bias=False,
                    indice_key='subm1'), )
            block = self.pre_act_block
        else:
            norm_name, norm_layer = build_norm_layer(norm_cfg, 16)
            self.conv_input = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    16,
                    3,
                    padding=1,
                    bias=False,
                    indice_key='subm1'),
                norm_layer,
                nn.ReLU(),
            )
            block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_cfg=norm_cfg, padding=1,
                  indice_key='subm1'), )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(
                16,
                32,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key='spconv2',
                conv_type='spconv'),
            block(32, 32, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(
                32,
                64,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key='spconv3',
                conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(
                64,
                64,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=(0, 1, 1),
                indice_key='spconv4',
                conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )

        norm_name, norm_layer = build_norm_layer(norm_cfg, 128)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(
                128,
                128, (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                bias=False,
                indice_key='spconv_down2'),
            norm_layer,
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx]
        :param batch_size:
        :return:
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

    def pre_act_block(self,
                      in_channels,
                      out_channels,
                      kernel_size,
                      indice_key=None,
                      stride=1,
                      padding=0,
                      conv_type='subm',
                      norm_cfg=None):
        norm_name, norm_layer = build_norm_layer(norm_cfg, in_channels)
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                norm_layer,
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                norm_layer,
                nn.ReLU(inplace=True),
                spconv.SparseConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key),
            )
        else:
            raise NotImplementedError
        return m

    def post_act_block(self,
                       in_channels,
                       out_channels,
                       kernel_size,
                       indice_key,
                       stride=1,
                       padding=0,
                       conv_type='subm',
                       norm_cfg=None):
        norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    bias=False,
                    indice_key=indice_key),
                norm_layer,
                nn.ReLU(inplace=True),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key),
                norm_layer,
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError
        return m
