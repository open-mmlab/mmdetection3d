import torch
import torch.nn as nn

import mmdet3d.ops.spconv as spconv
from mmdet.ops import build_norm_layer
from ..registry import MIDDLE_ENCODERS
from .sparse_block_utils import SparseBasicBlock


@MIDDLE_ENCODERS.register_module
class SparseUnetV2(nn.Module):

    def __init__(self,
                 in_channels,
                 output_shape,
                 pre_act,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        """SparseUnet for PartA^2

        Args:
            in_channels (int): the number of input channels
            output_shape (list[int]): the shape of output tensor
            pre_act (bool): use pre_act_block or post_act_block
            norm_cfg (dict): normalize layer config
        """
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
                64,
                128, (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                bias=False,
                indice_key='spconv_down2'),
            norm_layer,
            nn.ReLU(),
        )

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(
            64, 64, indice_key='subm4', norm_cfg=norm_cfg)
        self.conv_up_m4 = block(
            128, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4')
        self.inv_conv4 = block(
            64,
            64,
            3,
            norm_cfg=norm_cfg,
            indice_key='spconv4',
            conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(
            64, 64, indice_key='subm3', norm_cfg=norm_cfg)
        self.conv_up_m3 = block(
            128, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3')
        self.inv_conv3 = block(
            64,
            32,
            3,
            norm_cfg=norm_cfg,
            indice_key='spconv3',
            conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(
            32, 32, indice_key='subm2', norm_cfg=norm_cfg)
        self.conv_up_m2 = block(
            64, 32, 3, norm_cfg=norm_cfg, indice_key='subm2')
        self.inv_conv2 = block(
            32,
            16,
            3,
            norm_cfg=norm_cfg,
            indice_key='spconv2',
            conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(
            16, 16, indice_key='subm1', norm_cfg=norm_cfg)
        self.conv_up_m1 = block(
            32, 16, 3, norm_cfg=norm_cfg, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1'))

        self.seg_cls_layer = nn.Linear(16, 1, bias=True)
        self.seg_reg_layer = nn.Linear(16, 3, bias=True)

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseUnetV2

        Args:
            voxel_features (torch.float32): shape [N, C]
            coors (torch.int32): shape [N, 4](batch_idx, z_idx, y_idx, x_idx)
            batch_size (int): batch size

        Returns:
            dict: backbone features
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

        ret = {'spatial_features': spatial_features}

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4,
                                      self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3,
                                      self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2,
                                      self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1,
                                      self.conv_up_m1, self.conv5)

        seg_features = x_up1.features

        seg_cls_preds = self.seg_cls_layer(seg_features)  # (N, 1)
        seg_reg_preds = self.seg_reg_layer(seg_features)  # (N, 3)

        ret.update({
            'u_seg_preds': seg_cls_preds,
            'u_reg_preds': seg_reg_preds,
            'seg_features': seg_features
        })

        return ret

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        """Forward of upsample and residual block.

        Args:
            x_lateral (SparseConvTensor): lateral tensor
            x_bottom (SparseConvTensor): tensor from bottom layer
            conv_t (SparseBasicBlock): convolution for lateral tensor
            conv_m (SparseSequential): convolution for merging features
            conv_inv (SparseSequential): convolution for upsampling

        Returns:
            SparseConvTensor: upsampled feature
        """
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """Channel reduction for element-wise add.

        Args:
            x (SparseConvTensor): x.features (N, C1)
            out_channels (int): the number of channel after reduction

        Returns:
            SparseConvTensor: channel reduced feature
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels %
                out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def pre_act_block(self,
                      in_channels,
                      out_channels,
                      kernel_size,
                      indice_key=None,
                      stride=1,
                      padding=0,
                      conv_type='subm',
                      norm_cfg=None):
        """Make pre activate sparse convolution block.

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of out channels
            kernel_size (int): kernel size of convolution
            indice_key (str): the indice key used for sparse tensor
            stride (int): the stride of convolution
            padding (int or list[int]): the padding number of input
            conv_type (str): conv type in 'subm', 'spconv' or 'inverseconv'
            norm_cfg (dict): normal layer configs

        Returns:
            spconv.SparseSequential: pre activate sparse convolution block.
        """
        assert conv_type in ['subm', 'spconv', 'inverseconv']

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
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                norm_layer,
                nn.ReLU(inplace=True),
                spconv.SparseInverseConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
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
        """Make post activate sparse convolution block.

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of out channels
            kernel_size (int): kernel size of convolution
            indice_key (str): the indice key used for sparse tensor
            stride (int): the stride of convolution
            padding (int or list[int]): the padding number of input
            conv_type (str): conv type in 'subm', 'spconv' or 'inverseconv'
            norm_cfg (dict[str]): normal layer configs

        Returns:
            spconv.SparseSequential: post activate sparse convolution block.
        """
        assert conv_type in ['subm', 'spconv', 'inverseconv']

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
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    bias=False,
                    indice_key=indice_key),
                norm_layer,
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError
        return m
