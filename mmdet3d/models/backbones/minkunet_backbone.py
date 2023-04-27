# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers import (TorchSparseBasicBlock,
                                   TorchSparseBottleneck,
                                   TorchSparseConvModule)
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import ConfigType, OptMultiConfig

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse.tensor import PointTensor, SparseTensor
else:
    SparseTensor = None


@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        block_type (str): Type of block in encoder and decoder.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        norm_cfg (dict or :obj:`ConfigDict`): Config of normalization.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 block_type: str = 'basicblock',
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 norm_cfg: ConfigType = dict(type='TorchSparseBN'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        self.num_stages = num_stages
        self.conv_input = nn.Sequential(
            TorchSparseConvModule(
                in_channels, base_channels, kernel_size=3, norm_cfg=norm_cfg),
            TorchSparseConvModule(
                base_channels, base_channels, kernel_size=3,
                norm_cfg=norm_cfg))

        if block_type == 'basicblock':
            block = TorchSparseBasicBlock
        elif block_type == 'bottleneck':
            block = TorchSparseBottleneck
        else:
            raise NotImplementedError(f'Unsppported block type: {block_type}')

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            self.encoder.append(
                nn.Sequential(
                    TorchSparseConvModule(
                        encoder_channels[i],
                        encoder_channels[i],
                        kernel_size=2,
                        stride=2,
                        norm_cfg=norm_cfg),
                    block(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=3,
                        norm_cfg=norm_cfg), *[
                            block(
                                encoder_channels[i + 1],
                                encoder_channels[i + 1],
                                kernel_size=3,
                                norm_cfg=norm_cfg)
                        ] * (encoder_blocks[i] - 1)))

            self.decoder.append(
                nn.ModuleList([
                    TorchSparseConvModule(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        transposed=True,
                        norm_cfg=norm_cfg),
                    nn.Sequential(
                        block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            kernel_size=3,
                            norm_cfg=norm_cfg), *[
                                block(
                                    decoder_channels[i + 1],
                                    decoder_channels[i + 1],
                                    kernel_size=3,
                                    norm_cfg=norm_cfg)
                            ] * (decoder_blocks[i] - 1))
                ]))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            SparseTensor: Backbone features.
        """
        x = torchsparse.SparseTensor(voxel_features, coors)
        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)
            x = torchsparse.cat((x, laterals[i]))
            x = decoder_layer[1](x)
            decoder_outs.append(x)

        return decoder_outs[-1].F


@MODELS.register_module()
class MinkUNetBackboneV2(MinkUNetBackbone):
    r"""MinkUNet backbone V2.

    refer to https://github.com/PJLab-ADG/PCSeg/blob/master/pcseg/model/segmentor/voxel/minkunet/minkunet.py

    """  # noqa: E501

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            SparseTensor: Backbone features.
        """
        voxels = SparseTensor(voxel_features, coors)
        points = PointTensor(voxels.F, voxels.C.float())

        voxels = initial_voxelize(points)
        voxels = self.conv_input(voxels)
        points = voxel_to_point(voxels, points)

        laterals = [voxels]
        for encoder_layer in self.encoder:
            voxels = encoder_layer(voxels)
            laterals.append(voxels)
        laterals = laterals[:-1][::-1]
        points = voxel_to_point(voxels, points)
        outputs = [points.F]

        for i, decoder_layer in enumerate(self.decoder):
            voxels = decoder_layer[0](voxels)
            voxels = torchsparse.cat((voxels, laterals[i]))
            voxels = decoder_layer[1](voxels)
            if i % 2 == 1:
                points = voxel_to_point(voxels, points)
                outputs.append(points.F)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def initial_voxelize(points: PointTensor) -> SparseTensor:
    """Voxelization again based on input PointTensor.

    Args:
        points (PointTensor): Input points after voxelization.

    Returns:
        SparseTensor: New voxels.
    """
    pc_hash = F.sphash(torch.floor(points.C).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(points.C), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(points.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    points.additional_features['idx_query'][1] = idx_query
    points.additional_features['counts'][1] = counts
    return new_tensor


def voxel_to_point(voxels: SparseTensor,
                   points: PointTensor,
                   point_transform: Optional[nn.Module] = None,
                   nearest: bool = False) -> PointTensor:
    """Feed voxel features to points.

    Args:
        voxels (SparseTensor): Input voxels.
        points (PointTensor): Input points.
        point_transform (nn.Module, optional): Point transform module
            for input point features. Defaults to None.
        nearest (bool): Whether to use nearest neighbor interpolation.
            Defaults to False.

    Returns:
        PointTensor: Points with new features.
    """
    if points.idx_query is None or points.weights is None or \
            points.idx_query.get(voxels.s) is None or \
            points.weights.get(voxels.s) is None:
        offsets = get_kernel_offsets(2, voxels.s, 1, device=points.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(points.C[:, :3] / voxels.s[0]).int() * voxels.s[0],
                points.C[:, -1].int().view(-1, 1)
            ], 1), offsets)
        pc_hash = F.sphash(voxels.C.to(points.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(
            points.C, idx_query, scale=voxels.s[0]).transpose(0,
                                                              1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_features = F.spdevoxelize(voxels.F, idx_query, weights)
        new_tensor = PointTensor(
            new_features,
            points.C,
            idx_query=points.idx_query,
            weights=points.weights)
        new_tensor.additional_features = points.additional_features
        new_tensor.idx_query[voxels.s] = idx_query
        new_tensor.weights[voxels.s] = weights
        points.idx_query[voxels.s] = idx_query
        points.weights[voxels.s] = weights
    else:
        new_features = F.spdevoxelize(voxels.F, points.idx_query.get(voxels.s),
                                      points.weights.get(voxels.s))
        new_tensor = PointTensor(
            new_features,
            points.C,
            idx_query=points.idx_query,
            weights=points.weights)
        new_tensor.additional_features = points.additional_features

    if point_transform is not None:
        new_tensor.F = new_tensor.F + point_transform(points.F)

    return new_tensor


def point_to_voxel(voxels: SparseTensor, points: PointTensor) -> SparseTensor:
    """Feed point features to voxels.

    Args:
        voxels (SparseTensor): Input voxels.
        points (PointTensor): Input points.

    Returns:
        SparseTensor: Voxels with new features.
    """
    if points.additional_features is None or \
            points.additional_features.get('idx_query') is None or \
            points.additional_features['idx_query'].get(voxels.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(points.C[:, :3] / voxels.s[0]).int() * voxels.s[0],
                points.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(voxels.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), voxels.C.shape[0])
        points.additional_features['idx_query'][voxels.s] = idx_query
        points.additional_features['counts'][voxels.s] = counts
    else:
        idx_query = points.additional_features['idx_query'][voxels.s]
        counts = points.additional_features['counts'][voxels.s]

    inserted_features = F.spvoxelize(points.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_features, voxels.C, voxels.s)
    new_tensor.cmaps = voxels.cmaps
    new_tensor.kmaps = voxels.kmaps

    return new_tensor
