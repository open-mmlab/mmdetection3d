# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List, Optional

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from spconv.pytorch import SparseConvTensor
from torch import Tensor, nn

from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                SparseBottleneck,
                                                make_sparse_convmodule,
                                                replace_feature)
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)
from mmdet3d.utils import OptMultiConfig

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse.tensor import PointTensor, SparseTensor
else:
    PointTensor = SparseTensor = None

# def build_sparse_module(cfg: Dict, *args, **kwargs) -> nn.Module:
#     """Build sparse module.

#     Args:
#         cfg (None or dict): The conv layer config, which should contain:
#             - type (str): Layer type.
#             - layer args: Args needed to instantiate an conv layer.
#         args (argument list): Arguments passed to the `__init__`
#             method of the corresponding conv layer.
#         kwargs (keyword arguments): Keyword arguments passed to the
#            `__init__`
#             method of the corresponding conv layer.

#     Returns:
#         nn.Module: Created conv layer.
#     """
#     if not isinstance(cfg, dict):
#         raise TypeError('cfg must be a dict')
#     if 'type' not in cfg:
#         raise KeyError('the cfg dict must contain the key "type"')
#     cfg_ = cfg.copy()

#     module_type = cfg_.pop('type')
#     if module_type == "SparseConvModule":
#         module = make_sparse_convmodule(*args, **kwargs, **cfg_)
#         return module

#     with MODELS.switch_scope_and_registry(None) as registry:
#         sparse_module = registry.get(module_type)
#     if sparse_module is None:
#         raise KeyError(f'Cannot find {sparse_module} in
#           registry under scope '
#                        f'name {registry.scope}')
#     module = sparse_module(*args, **kwargs, **cfg_)
#     return module


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
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 block_type: str = 'basic',
                 sparseconv_backends: str = 'torchsparse',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backends in [
            'torchsparse', 'spconv', 'minkowski'
        ], f'sparseconv backend: {sparseconv_backends} not supported.'
        self.num_stages = num_stages
        self.sparseconv_backends = sparseconv_backends
        if sparseconv_backends == 'torchsparse':
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            decoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' \
                else TorchSparseBottleneck
            residual_branch = None  # for torchsparse
        elif sparseconv_backends == 'spconv':
            input_conv = partial(
                make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseConv3d')
            decoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseInverseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' \
                else SparseBottleneck
            residual_branch = partial(
                make_sparse_convmodule, conv_type='SubMConv3d')

        self.conv_input = nn.Sequential(
            input_conv(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'),
            input_conv(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(
                    encoder_channels[i],
                    encoder_channels[i],
                    kernel_size=2,
                    stride=2,
                    indice_key=f'spconv{i+1}')
            ]
            for j in range(encoder_blocks[i]):
                if j == 0:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i],
                            encoder_channels[i + 1],
                            downsample=residual_branch(
                                encoder_channels[i],
                                encoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'))
                else:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i + 1],
                            encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

            decoder_layer = [
                decoder_conv(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    transposed=True,
                    indice_key=f'spconv{num_stages-i}')
            ]
            for j in range(decoder_blocks[i]):
                if j == 0:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            downsample=residual_branch(
                                decoder_channels[i + 1] +
                                encoder_channels[-2 - i],
                                decoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{num_stages-i}'))
                else:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            indice_key=f'subm{num_stages-i}'))
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            SparseTensor: Backbone features.
        """
        if self.sparseconv_backends == 'torchsparse':
            x = SparseTensor(voxel_features, coors)
        elif self.sparseconv_backends == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = coors[-1, 0] + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                 batch_size)
        elif self.sparseconv_backends == 'minkowski':
            pass

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)
            if self.sparseconv_backends == 'torchsparse':
                x = torchsparse.cat((x, laterals[i]))
            elif self.sparseconv_backends == 'spconv':
                x = replace_feature(x, torch.cat((x, laterals[i])))
            elif self.sparseconv_backends == 'minkowski':
                pass
            x = decoder_layer[1](x)
            decoder_outs.append(x)

        return decoder_outs[-1]


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
        output_features = [points.F]

        for i, decoder_layer in enumerate(self.decoder):
            voxels = decoder_layer[0](voxels)
            voxels = torchsparse.cat((voxels, laterals[i]))
            voxels = decoder_layer[1](voxels)
            if i % 2 == 1:
                points = voxel_to_point(voxels, points)
                output_features.append(points.F)

        points.F = torch.cat(output_features, dim=1)
        return points


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
