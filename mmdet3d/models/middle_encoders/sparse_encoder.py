# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.ops import points_in_boxes_all, three_interpolate, three_nn
from mmdet.models.losses import sigmoid_focal_loss, smooth_l1_loss
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


@MODELS.register_module()
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
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
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
                 block_type='conv_module',
                 return_middle_feats=False):
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
        self.return_middle_feats = return_middle_feats
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

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
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

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
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


@MODELS.register_module()
class SparseEncoderSASSD(SparseEncoder):
    r"""Sparse encoder for `SASSD <https://github.com/skyhehe123/SA-SSD>`_

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
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels: int,
                 sparse_shape: List[int],
                 order: Tuple[str] = ('conv', 'norm', 'act'),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels: int = 16,
                 output_channels: int = 128,
                 encoder_channels: Tuple[tuple] = ((16, ), (32, 32, 32),
                                                   (64, 64, 64), (64, 64, 64)),
                 encoder_paddings: Tuple[tuple] = ((1, ), (1, 1, 1), (1, 1, 1),
                                                   ((0, 1, 1), 1, 1)),
                 block_type: str = 'conv_module'):
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

    def forward(self,
                voxel_features: Tensor,
                coors: Tensor,
                batch_size: Tensor,
                test_mode: bool = False) -> Tuple[Tensor, tuple]:
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.
            test_mode (bool, optional): Whether in test mode.
                Defaults to False.

        Returns:
            Tensor: Backbone features.
            tuple[torch.Tensor]: Mean feature value of the points,
                Classification result of the points,
                Regression offsets of the points.
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

        if test_mode:
            return spatial_features, None

        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        # auxiliary network
        p0 = self.make_auxiliary_points(
            encode_features[0],
            points_mean,
            offset=(0, -40., -3.),
            voxel_size=(.1, .1, .2))

        p1 = self.make_auxiliary_points(
            encode_features[1],
            points_mean,
            offset=(0, -40., -3.),
            voxel_size=(.2, .2, .4))

        p2 = self.make_auxiliary_points(
            encode_features[2],
            points_mean,
            offset=(0, -40., -3.),
            voxel_size=(.4, .4, .8))

        pointwise = torch.cat([p0, p1, p2], dim=-1)
        pointwise = self.point_fc(pointwise)
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)
        point_misc = (points_mean, point_cls, point_reg)

        return spatial_features, point_misc

    def get_auxiliary_targets(self,
                              points_feats: Tensor,
                              gt_bboxes_3d: List[BaseInstance3DBoxes],
                              enlarge: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Get auxiliary target.

        Args:
            points_feats (torch.Tensor): Mean features of the points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):  Ground truth
                boxes for each sample.
            enlarge (float, optional): Enlaged scale. Defaults to 1.0.

        Returns:
            tuple[torch.Tensor]: Label of the points and
                center offsets of the points.
        """
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_bboxes_3d)):
            boxes3d = gt_bboxes_3d[i].tensor.detach().clone()
            idx = torch.nonzero(points_feats[:, 0] == i).view(-1)
            point_xyz = points_feats[idx, 1:].detach().clone()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = self.calculate_pts_offsets(
                point_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()
            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets)
        pts_labels = torch.cat(pts_labels).to(center_offsets.device)

        return pts_labels, center_offsets

    def calculate_pts_offsets(self, points: Tensor,
                              bboxes_3d: Tensor) -> Tuple[Tensor, Tensor]:
        """Find all boxes in which each point is, as well as the offsets from
        the box centers.

        Args:
            points (torch.Tensor): [M, 3], [x, y, z] in LiDAR coordinate
            bboxes_3d (torch.Tensor): [T, 7],
                num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
                (x, y, z) is the bottom center.

        Returns:
            tuple[torch.Tensor]: Point indices of boxes with the shape of
                (T, M). Default background = 0.
                And offsets from the box centers of points,
                if it belows to the box, with the shape of (M, 3).
                Default background = 0.
        """
        boxes_num = len(bboxes_3d)
        pts_num = len(points)

        box_indices = points_in_boxes_all(points[None, ...], bboxes_3d[None,
                                                                       ...])
        pts_indices = box_indices.squeeze(0).transpose(0, 1)
        center_offsets = torch.zeros_like(points).to(points.device)

        for i in range(boxes_num):
            for j in range(pts_num):
                if pts_indices[i][j] == 1:
                    center_offsets[j][0] = points[j][0] - bboxes_3d[i][0]
                    center_offsets[j][1] = points[j][1] - bboxes_3d[i][1]
                    center_offsets[j][2] = (
                        points[j][2] -
                        (bboxes_3d[i][2] + bboxes_3d[i][2] / 2.0))
        return pts_indices, center_offsets

    def aux_loss(self, points: Tensor, point_cls: Tensor, point_reg: Tensor,
                 gt_bboxes_3d: Tensor) -> dict:
        """Calculate auxiliary loss.

        Args:
            points (torch.Tensor): Mean feature value of the points.
            point_cls (torch.Tensor): Classification result of the points.
            point_reg (torch.Tensor): Regression offsets of the points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.

        Returns:
            dict: Auxiliary loss.
        """
        num_boxes = len(gt_bboxes_3d)
        pts_labels, center_targets = self.get_auxiliary_targets(
            points, gt_bboxes_3d)

        rpn_cls_target = pts_labels.long()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum().clamp(min=1.0)

        cls_weights = pos + neg
        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = sigmoid_focal_loss(
            point_cls,
            rpn_cls_target,
            weight=cls_weights,
            avg_factor=pos_normalizer)

        aux_loss_cls /= num_boxes

        weight = reg_weights[..., None]
        aux_loss_reg = smooth_l1_loss(point_reg, center_targets, beta=1 / 9.)
        aux_loss_reg = torch.sum(aux_loss_reg * weight)[None]
        aux_loss_reg /= num_boxes

        aux_loss_cls, aux_loss_reg = [aux_loss_cls], [aux_loss_reg]

        return dict(aux_loss_cls=aux_loss_cls, aux_loss_reg=aux_loss_reg)

    def make_auxiliary_points(
        self,
        source_tensor: Tensor,
        target: Tensor,
        offset: Tuple = (0., -40., -3.),
        voxel_size: Tuple = (.05, .05, .1)
    ) -> Tensor:
        """Make auxiliary points for loss computation.

        Args:
            source_tensor (torch.Tensor): (M, C) features to be propigated.
            target (torch.Tensor): (N, 4) bxyz positions of the
                target features.
            offset (tuple[float], optional): Voxelization offset.
                Defaults to (0., -40., -3.)
            voxel_size (tuple[float], optional): Voxelization size.
                Defaults to (.05, .05, .1)

        Returns:
            torch.Tensor: (N, C) tensor of the features of the target features.
        """
        # Tansfer tensor to points
        source = source_tensor.indices.float()
        offset = torch.Tensor(offset).to(source.device)
        voxel_size = torch.Tensor(voxel_size).to(source.device)
        source[:, 1:] = (
            source[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size)

        source_feats = source_tensor.features[None, ...].transpose(1, 2)

        # Interplate auxiliary points
        dist, idx = three_nn(target[None, ...], source[None, ...])
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        new_features = three_interpolate(source_feats.contiguous(), idx,
                                         weight)

        return new_features.squeeze(0).transpose(0, 1)
