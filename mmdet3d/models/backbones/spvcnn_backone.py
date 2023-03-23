# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import OptMultiConfig
from .minkunet_backbone import MinkUNetBackbone

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse.tensor import PointTensor, SparseTensor
else:
    PointTensor = SparseTensor = None


@MODELS.register_module()
class SPVCNNBackbone(MinkUNetBackbone):
    """SPVCNN backbone with torchsparse backend.

    More details can be found in `paper <https://arxiv.org/abs/2007.16100>`_ .

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        drop_ratio (float): Dropout ratio of voxel features. Defaults to 0.3.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 encoder_channels: Sequence[int] = [32, 64, 128, 256],
                 decoder_channels: Sequence[int] = [256, 128, 96, 96],
                 num_stages: int = 4,
                 drop_ratio: float = 0.3,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_stages=num_stages,
            init_cfg=init_cfg)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_channels, encoder_channels[-1]),
                nn.BatchNorm1d(encoder_channels[-1]), nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(encoder_channels[-1], decoder_channels[2]),
                nn.BatchNorm1d(decoder_channels[2]), nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(decoder_channels[2], decoder_channels[4]),
                nn.BatchNorm1d(decoder_channels[4]), nn.ReLU(True))
        ])
        self.dropout = nn.Dropout(drop_ratio, True)

    def forward(self, voxel_features: Tensor, coors: Tensor) -> PointTensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            PointTensor: Backbone features.
        """
        voxels = SparseTensor(voxel_features, coors)
        points = PointTensor(voxels.F, voxels.C.float())
        voxels = self.initial_voxelize(points)

        voxels = self.conv_input(voxels)
        points = self.voxel_to_point(voxels, points)
        voxels = self.point_to_voxel(voxels, points)
        laterals = [voxels]
        for encoder in self.encoder:
            voxels = encoder(voxels)
            laterals.append(voxels)
        laterals = laterals[:-1][::-1]

        points = self.voxel_to_point(voxels, points, self.point_transforms[0])
        voxels = self.point_to_voxel(voxels, points)
        voxels.F = self.dropout(voxels.F)

        decoder_outs = []
        for i, decoder in enumerate(self.decoder):
            voxels = decoder[0](voxels)
            voxels = torchsparse.cat((voxels, laterals[i]))
            voxels = decoder[1](voxels)
            decoder_outs.append(voxels)
            if i == 1:
                points = self.voxel_to_point(voxels, points,
                                             self.point_transforms[1])
                voxels = self.point_to_voxel(voxels, points)
                voxels.F = self.dropout(voxels.F)

        points = self.voxel_to_point(voxels, points, self.point_transforms[2])
        return points

    def initial_voxelize(self, points: PointTensor) -> SparseTensor:
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

        inserted_coords = F.spvoxelize(
            torch.floor(points.C), idx_query, counts)
        inserted_coords = torch.round(inserted_coords).int()
        inserted_feat = F.spvoxelize(points.F, idx_query, counts)

        new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
        new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
        points.additional_features['idx_query'][1] = idx_query
        points.additional_features['counts'][1] = counts
        return new_tensor

    def voxel_to_point(self,
                       voxels: SparseTensor,
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
            offsets = get_kernel_offsets(
                2, voxels.s, 1, device=points.F.device)
            old_hash = F.sphash(
                torch.cat([
                    torch.floor(points.C[:, :3] / voxels.s[0]).int() *
                    voxels.s[0], points.C[:, -1].int().view(-1, 1)
                ], 1), offsets)
            pc_hash = F.sphash(voxels.C.to(points.F.device))
            idx_query = F.sphashquery(old_hash, pc_hash)
            weights = F.calc_ti_weights(
                points.C, idx_query,
                scale=voxels.s[0]).transpose(0, 1).contiguous()
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
            new_features = F.spdevoxelize(voxels.F,
                                          points.idx_query.get(voxels.s),
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

    def point_to_voxel(self, voxels: SparseTensor,
                       points: PointTensor) -> SparseTensor:
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
                    torch.floor(points.C[:, :3] / voxels.s[0]).int() *
                    voxels.s[0], points.C[:, -1].int().view(-1, 1)
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
