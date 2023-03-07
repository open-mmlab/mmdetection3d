# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import OptMultiConfig
from .minkunet_backbone import MinkUNetBackbone

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse.tensor import PointTensor, SparseTensor
else:
    SparseTensor = None


@MODELS.register_module()
class SPVCNNBackbone(MinkUNetBackbone):
    """MinkUNet backbone is the implementation of `4D Spatio-Temporal ConvNets.

    <https://arxiv.org/abs/1904.08755>` with torchsparse backend.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Defaults to 64.
        enc_channels (tuple[int]): Convolutional channels of each encode block.
        dec_channels (tuple[int]): Convolutional channels of each decode block.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 enc_channels: Sequence[int] = [32, 64, 128, 256],
                 dec_channels: Sequence[int] = [256, 128, 96, 96],
                 num_stages: int = 4,
                 drop_ratio: float = 0.3,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            num_stages=num_stages,
            init_cfg=init_cfg)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_channels, enc_channels[-1]),
                nn.BatchNorm1d(enc_channels[-1]), nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(enc_channels[-1], dec_channels[2]),
                nn.BatchNorm1d(dec_channels[2]), nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(dec_channels[2], dec_channels[4]),
                nn.BatchNorm1d(dec_channels[4]), nn.ReLU(True))
        ])
        self.dropout = nn.Dropout(drop_ratio, True)

    def forward(self, voxel_features: Tensor, coors: Tensor) -> PointTensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            SparseTensor: backbone features.
        """
        x = SparseTensor(voxel_features, coors)
        z = PointTensor(x.F, x.C.float())
        x = self.initial_voxelize(z)

        x = self.conv_input(x)
        z = self.voxel_to_point(x, z, nearest=False)
        x = self.point_to_voxel(x, z)
        laterals = [x]
        for enc in self.encoder:
            x = enc(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        z_new = self.voxel_to_point(x, z)
        z_new.F = z_new.F + self.point_transforms[0](z.F)
        z = z_new
        x = self.point_to_voxel(x, z)
        x.F = self.dropout(x.F)

        dec_outs = []
        for i, dec in enumerate(self.decoder):
            x = dec[0](x)
            x = torchsparse.cat((x, laterals[i]))
            x = dec[1](x)
            dec_outs.append(x)
            if i == 1:
                z_new = self.voxel_to_point(x, z)
                z_new.F = z_new.F + self.point_transforms[1](z.F)
                z = z_new
                x = self.point_to_voxel(x, z)
                x.F = self.dropout(x.F)

        z_new = self.voxel_to_point(x, z)
        z_new.F = z_new.F + self.point_transforms[2](z.F)
        # x = self.point_to_voxel(x, z)

        return z_new

    def initial_voxelize(self, z):

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

    # x: SparseTensor, z: PointTensor
    # return: SparseTensor
    def point_to_voxel(self, x, z):
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

    # x: SparseTensor, z: PointTensor
    # return: PointTensor
    def voxel_to_point(self, x, z, nearest=False):
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

        return new_tensor
