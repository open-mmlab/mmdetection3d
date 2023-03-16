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
        drop_ratio: Dropout ratio of voxel features. Defaults to 0.3.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
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
        x = SparseTensor(voxel_features, coors)
        z = PointTensor(x.F, x.C.float())
        x = self.initial_voxelize(z)

        x = self.conv_input(x)
        z = self.voxel_to_point(x, z)
        x = self.point_to_voxel(x, z)
        laterals = [x]
        for encoder in self.encoder:
            x = encoder(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        z = self.voxel_to_point(x, z, self.point_transforms[0])
        x = self.point_to_voxel(x, z)
        x.F = self.dropout(x.F)

        decoder_outs = []
        for i, decoder in enumerate(self.decoder):
            x = decoder[0](x)
            x = torchsparse.cat((x, laterals[i]))
            x = decoder[1](x)
            decoder_outs.append(x)
            if i == 1:
                z = self.voxel_to_point(x, z, self.point_transforms[1])
                x = self.point_to_voxel(x, z)
                x.F = self.dropout(x.F)

        z = self.voxel_to_point(x, z, self.point_transforms[2])
        return z

    def initial_voxelize(self, z: PointTensor) -> SparseTensor:
        """Voxelization again based on input PointTensor.

        Args:
            z (PointTensor): Input points after voxelization.

        Returns:
            SparseTensor: New voxels.
        """
        pc_hash = F.sphash(torch.floor(z.C).int())
        sparse_hash = torch.unique(pc_hash)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), len(sparse_hash))

        inserted_coords = F.spvoxelize(torch.floor(z.C), idx_query, counts)
        inserted_coords = torch.round(inserted_coords).int()
        inserted_feat = F.spvoxelize(z.F, idx_query, counts)

        new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
        new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
        z.additional_features['idx_query'][1] = idx_query
        z.additional_features['counts'][1] = counts
        return new_tensor

    def voxel_to_point(self,
                       x: SparseTensor,
                       z: PointTensor,
                       point_transform: Optional[nn.Module] = None,
                       nearest: bool = False) -> PointTensor:
        """Feed voxel features to points.

        Args:
            x (SparseTensor): Input voxels.
            z (PointTensor): Input points.
            point_transform: (nn.Module, optional): Point transform module
                for input point features. Defaults to None.
            nearest: Whether to use nearest neighbor interpolation.
                Defaults to False.

        Returns:
            PointTensor: Points with new features.
        """
        if z.idx_query is None or z.weights is None or z.idx_query.get(
                x.s) is None or z.weights.get(x.s) is None:
            off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
            old_hash = F.sphash(
                torch.cat([
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1)
                ], 1), off)
            pc_hash = F.sphash(x.C.to(z.F.device))
            idx_query = F.sphashquery(old_hash, pc_hash)
            weights = F.calc_ti_weights(
                z.C, idx_query, scale=x.s[0]).transpose(0, 1).contiguous()
            idx_query = idx_query.transpose(0, 1).contiguous()
            if nearest:
                weights[:, 1:] = 0.
                idx_query[:, 1:] = -1
            new_feat = F.spdevoxelize(x.F, idx_query, weights)
            new_tensor = PointTensor(
                new_feat, z.C, idx_query=z.idx_query, weights=z.weights)
            new_tensor.additional_features = z.additional_features
            new_tensor.idx_query[x.s] = idx_query
            new_tensor.weights[x.s] = weights
            z.idx_query[x.s] = idx_query
            z.weights[x.s] = weights
        else:
            new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s),
                                      z.weights.get(x.s))
            new_tensor = PointTensor(
                new_feat, z.C, idx_query=z.idx_query, weights=z.weights)
            new_tensor.additional_features = z.additional_features

        if point_transform is not None:
            new_tensor.F = new_tensor.F + point_transform(z.F)

        return new_tensor

    def point_to_voxel(self, x: SparseTensor, z: PointTensor) -> SparseTensor:
        """Feed point features to voxels.

        Args:
            x (SparseTensor): Input voxels.
            z (PointTensor): Input points.

        Returns:
            SparseTensor: Voxels with new features.
        """
        if z.additional_features is None or z.additional_features.get(
                'idx_query') is None or z.additional_features['idx_query'].get(
                    x.s) is None:
            pc_hash = F.sphash(
                torch.cat([
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1)
                ], 1))
            sparse_hash = F.sphash(x.C)
            idx_query = F.sphashquery(pc_hash, sparse_hash)
            counts = F.spcount(idx_query.int(), x.C.shape[0])
            z.additional_features['idx_query'][x.s] = idx_query
            z.additional_features['counts'][x.s] = counts
        else:
            idx_query = z.additional_features['idx_query'][x.s]
            counts = z.additional_features['counts'][x.s]

        inserted_feat = F.spvoxelize(z.F, idx_query, counts)
        new_tensor = SparseTensor(inserted_feat, x.C, x.s)
        new_tensor.cmaps = x.cmaps
        new_tensor.kmaps = x.kmaps

        return new_tensor
