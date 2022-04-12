# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import (SparseConvTensor, SparseSequential,
                      three_interpolate, three_nn)
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels


@MIDDLE_ENCODERS.register_module()
class SparseEncoderSASSD(SparseEncoder):
    r"""Sparse encoder for SASSD.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module'):
        super(SparseEncoderSASSD, self).__init__(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            order=order,
            norm_cfg=norm_cfg,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            block_type=block_type)

        self.point_fc = nn.Linear(112, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size, test_mode=False):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.
            test_mode (bool): Whether in test mode.

        Returns:
            dict: Backbone features.
            tuple[torch.Tensor]: Mean feature value of the points,
                classificaion result of the points,
                regression offsets of the points.

        """

        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape,
                                           batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        if test_mode:
            return spatial_features, None

        # auxiliary network
        p0 = self.make_auxiliary_points(encode_features[0], points_mean,
          (0, -40., -3.), voxel_size=(.1, .1, .2))

        p1 = self.make_auxiliary_points(encode_features[1], points_mean,
          (0, -40., -3.), voxel_size=(.2, .2, .4))

        p2 = self.make_auxiliary_points(encode_features[2], points_mean,
          (0, -40., -3.), voxel_size=(.4, .4, .8))

        pointwise = torch.cat([p0, p1, p2], dim=-1)
        pointwise = self.point_fc(pointwise)
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)

        return spatial_features, (points_mean, point_cls, point_reg)

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        """Build auxiliary target.

        Args:
            nxyz (torch.Tensor): Mean features of the points.
            gt_boxes3d (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            enlarge (int, optional): Enlage scale.

        Returns:
            torch.Tensor: Label of the poitns.
            torch.Tensor: Center offsets of the points.
        """
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].tensor.cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()
            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):
        """Calculate auxiliary loss.

        Args:
            points (torch.Tensor): Mean feature value of the points.
            point_cls (torch.Tensor): Classificaion result of the points.
            point_reg (torch.Tensor): Regression offsets of the points.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.

        Returns:
            dict: Backbone features.
        """
        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.long()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = sigmoid_focal_loss(
            point_cls,
            rpn_cls_target,
            weight=cls_weights)

        aux_loss_cls /= N

        weight = reg_weights[..., None]
        aux_loss_reg = smooth_l1_loss(point_reg, center_targets, beta=1/9.)
        aux_loss_reg = torch.sum(aux_loss_reg * weight)[None]
        aux_loss_reg /= N

        return dict(aux_loss_cls=aux_loss_cls, aux_loss_reg=aux_loss_reg)

    def make_auxiliary_points(self, source_tensor, target, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
        """
        :param source_tensor (Tensor): (m, C) tensor of features to be propigated
        :param target (Tensor): (n, 4) tensor of the bxyz positions of the unknown features
        :param offset (tuple): voxelization offset
        :param voxel_size (tuple): voxelization size
        :return:
            new_features: (n, C) tensor of the features of the target features
        """
        # Tansfer tensor to points
        source = source_tensor.indices.float()
        offset = torch.Tensor(offset).to(source.device)
        voxel_size = torch.Tensor(voxel_size).to(source.device)
        source[:, 1:] = (
            source[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size)

        source_feats = source_tensor.features

        # Interplate auxiliary points
        dist, idx = three_nn(target[None, ...], source[None, ...])
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        new_features = three_interpolate(source_feats[None, ...].transpose(1, 2), idx, weight)

        return new_features.squeeze(0).transpose(0, 1)